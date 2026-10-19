"""The model-kernel correction of the Roothaan step.

In the basis of the Roothaan orbitals of the extrapolated Fock matrix, the correction solves the linearized model
problem

    (Omega + K~) kappa = -g,    g = [K~ (D_R - D_bar)]_vo,    Omega_ai = e_a - e_i,

and rotates the occupied orbitals into the virtual ones by kappa.  K~ = 0 is plain DIIS; K~ = K, the exact response
kernel, is a Newton step on the DIIS model.  Omega is floored at omega_floor and max|kappa| is capped at max_rot.

    correct_lowrank  exact solve with the Woodbury identity (dense-store model)
    correct_cg       Omega-preconditioned conjugate gradients in the AO basis (any store); for large systems
    correct_hx       Hartree plus exact exchange in the same auxiliary basis, solved by MINRES
    correct_fock     the correction applied to the Fock matrix; valid for fractional (smeared) occupations
    correct_exact    the exact kernel (each product costs a response build); an upper bound, not a method
"""
import numpy as np
import scipy.linalg as sla
from scipy.optimize import brentq
from scipy.sparse.linalg import LinearOperator, minres


def _channels(mo_e, mo_c, mo_occ):
    """Per spin channel (e_occ, e_vir, C_occ, C_vir, occ_idx, vir_idx); restricted orbitals give one channel."""
    if np.ndim(mo_c) == 2:
        chans = [(mo_e, mo_c, mo_occ)]
    else:
        chans = [(mo_e[s], mo_c[s], mo_occ[s]) for s in range(2)]
    out = []
    for e, c, occ in chans:
        o, v = np.where(occ > 0)[0], np.where(occ == 0)[0]
        out.append((e[o], e[v], c[:, o], c[:, v], o, v))
    return out


def _rotate(c, o, v, kappa):
    no, nv = len(o), len(v)
    X = np.zeros((no + nv, no + nv))
    X[no:, :no] = kappa
    X[:no, no:] = -kappa.T
    cov = np.hstack([c[:, o], c[:, v]]) @ sla.expm(X)
    cnew = c.copy()
    cnew[:, o], cnew[:, v] = cov[:, :no], cov[:, no:]
    return cnew


def _rotate_svd(c, o, v, kappa):
    """Same as _rotate, with the exponential of the vo generator from the thin SVD of kappa (no expm)."""
    if kappa.size == 0:
        return c.copy()
    u, s, vt = np.linalg.svd(kappa, full_matrices=False)
    Co, Cv = c[:, o], c[:, v]
    CoV, CvU = Co @ vt.T, Cv @ u
    cnew = c.copy()
    cnew[:, o] = Co + (CoV * (np.cos(s) - 1.0)) @ vt + (CvU * np.sin(s)) @ vt
    cnew[:, v] = Cv + (CvU * (np.cos(s) - 1.0)) @ u.T - (CoV * np.sin(s)) @ u.T
    return cnew


def _cap(kappas, max_rot):
    m = max(np.abs(k).max() if k.size else 0.0 for k in kappas)
    if m > max_rot:
        kappas = [k * (max_rot / m) for k in kappas]
    return kappas, m


def fermi_occ(mo_e, nelec, sigma, restricted):
    """Fermi-Dirac occupations at fixed electron number per spin channel; returns (occ, entropy, f')."""

    def one(e, n, cap):
        def f(mu):
            return cap / (1.0 + np.exp(np.clip((e - mu) / sigma, -700, 700)))
        mu = brentq(lambda m: f(m).sum() - n, e.min() - 50 * sigma - 1.0, e.max() + 50 * sigma + 1.0, xtol=1e-14)
        occ = f(mu)
        x = np.clip(occ / cap, 1e-300, 1.0)
        y = np.clip(1.0 - occ / cap, 1e-300, 1.0)
        s = -cap * np.sum(x * np.log(x) + y * np.log(y))
        return occ, s, -occ * (1.0 - occ / cap) / sigma

    if restricted:
        return one(mo_e, sum(nelec), 2.0)
    out = [one(mo_e[k], nelec[k], 1.0) for k in range(2)]
    return np.array([o[0] for o in out]), sum(o[1] for o in out), np.array([o[2] for o in out])


def correct_lowrank(model, mo_e, mo_c, mo_occ, dD, omega_floor=1e-3, max_rot=0.3):
    """Exact solve with the Woodbury identity: the model is low-rank in the vo space, K~ = f B^T W B with f = 4
    (restricted) or 2 (unrestricted) and B_P,ai = (P|ai).  Returns (mo_coeff, max|kappa| before the cap, 0)."""
    restricted = np.ndim(mo_c) == 2
    f = 4.0 if restricted else 2.0
    chans = _channels(mo_e, mo_c, mo_occ)
    dD_total = dD if restricted else dD[0] + dD[1]
    u = model.W @ model.tvec(dD_total)
    A, D, g, shapes = [], [], [], []
    for eo, ev, Co, Cv, o, v in chans:
        B = model.vo(Co, Cv)
        A.append(B.reshape(B.shape[0], -1))
        g.append(np.einsum("Pvo,P->vo", B, u).ravel())
        D.append(np.maximum(ev[:, None] - eo[None, :], omega_floor).ravel())
        shapes.append((len(v), len(o)))
    A, D, r = np.hstack(A), np.concatenate(D), -np.concatenate(g)
    Dinv_r, AD = r / D, A / D
    fW = f * model.W
    M = np.eye(A.shape[0]) + (AD @ A.T) @ fW
    x = Dinv_r - AD.T @ (fW @ np.linalg.solve(M, A @ Dinv_r))
    kappas, i0 = [], 0
    for nv, no in shapes:
        kappas.append(x[i0:i0 + nv * no].reshape(nv, no))
        i0 += nv * no
    kappas, kmax = _cap(kappas, max_rot)
    if restricted:
        eo, ev, Co, Cv, o, v = chans[0]
        return _rotate(mo_c, o, v, kappas[0]), kmax, 0
    return np.array([_rotate(mo_c[s], chans[s][4], chans[s][5], kappas[s]) for s in range(2)]), kmax, 0


def correct_cg(model, mo_e, mo_c, mo_occ, dD, omega_floor=1e-3, max_rot=0.3, rtol=1e-3, maxiter=50):
    """The same correction, solved by conjugate gradients preconditioned with Omega.

    Each product applies K~ once in the AO basis, (Omega + K~) x = Omega x + [K~[c (Cv x Co^T + h.c.)]]_vo with
    c = 2 for restricted orbitals, so only model.apply is needed and the (naux, nvir, nocc) tensor B is never
    formed.  Returns (mo_coeff, max|kappa| before the cap, number of K~ applications)."""
    restricted = np.ndim(mo_c) == 2
    chans = _channels(mo_e, mo_c, mo_occ)
    c = 2.0 if restricted else 1.0
    om = [np.maximum(ch[1][:, None] - ch[0][None, :], omega_floor) for ch in chans]
    napply = [0]

    def kvo(V):
        return [ch[3].T @ (V @ ch[2]) for ch in chans]

    def kmat(xs):
        dm = sum((ch[3] @ x) @ ch[2].T for ch, x in zip(chans, xs))
        napply[0] += 1
        return kvo(model.apply(c * (dm + dm.T)))

    def dot(a, b):
        return sum(np.vdot(x, y) for x, y in zip(a, b))

    dD_total = dD if restricted else dD[0] + dD[1]
    napply[0] += 1
    b = [-g for g in kvo(model.apply(dD_total))]
    bnorm = np.sqrt(dot(b, b))
    x = [bi / o for bi, o in zip(b, om)]
    if bnorm > 0:
        r = [bi - o * xi - ki for bi, o, xi, ki in zip(b, om, x, kmat(x))]
        z = [ri / o for ri, o in zip(r, om)]
        p, rz = [zi.copy() for zi in z], dot(r, z)
        for _ in range(maxiter):
            if np.sqrt(dot(r, r)) < rtol * bnorm:
                break
            ap = [o * pi + ki for o, pi, ki in zip(om, p, kmat(p))]
            alpha = rz / dot(p, ap)
            x = [xi + alpha * pi for xi, pi in zip(x, p)]
            r = [ri - alpha * api for ri, api in zip(r, ap)]
            z = [ri / o for ri, o in zip(r, om)]
            rz, rz_old = dot(r, z), rz
            p = [zi + (rz / rz_old) * pi for zi, pi in zip(z, p)]
    kappas, kmax = _cap(x, max_rot)
    if restricted:
        return _rotate_svd(mo_c, chans[0][4], chans[0][5], kappas[0]), kmax, napply[0]
    return (np.array([_rotate_svd(mo_c[s], chans[s][4], chans[s][5], kappas[s]) for s in range(2)]),
            kmax, napply[0])


def hx_operator(model, chans, restricted, hyb, omega_floor=1e-3):
    """Pieces of correct_hx: kvec(xs) = K~ x per channel, rhs(dD) = [K~ dD]_vo, split, and the flat Omega."""
    W, T = model.W, model.T
    fh = 4.0 if restricted else 2.0
    blk = []
    for eo, ev, Co, Cv, o, v in chans:
        TCo, TCv = T @ Co, T @ Cv
        Bvo = np.einsum("mv,Pmo->Pvo", Cv, TCo, optimize=True)
        Bvv = np.einsum("mv,Pmw->Pvw", Cv, TCv, optimize=True)
        Boo = np.einsum("mo,Pmp->Pop", Co, TCo, optimize=True)
        blk.append(dict(Bvo=Bvo, Bvv=Bvv, Z=np.einsum("PQ,Qij->Pij", W, Boo), Y=np.einsum("PQ,Qai->Pai", W, Bvo),
                        TCo=TCo, Cv=Cv, om=np.maximum(ev[:, None] - eo[None, :], omega_floor)))
    shapes = [b["om"].shape for b in blk]
    sizes = [s[0] * s[1] for s in shapes]

    def split(x):
        return [p.reshape(s) for p, s in zip(np.split(x, np.cumsum(sizes)[:-1]), shapes)]

    def kvec(xs):
        t = sum(np.einsum("Pvo,vo->P", b["Bvo"], x) for b, x in zip(blk, xs))
        u = W @ t
        out = []
        for b, x in zip(blk, xs):
            h = fh * np.einsum("Pvo,P->vo", b["Bvo"], u)
            k = np.einsum("Pab,bj,Pji->ai", b["Bvv"], x, b["Z"], optimize=True) + \
                np.einsum("Paj,bj,Pbi->ai", b["Bvo"], x, b["Y"], optimize=True)
            out.append(h - hyb * k)
        return out

    def rhs(dD):
        dD_total = dD if restricted else dD[0] + dD[1]
        u0 = W @ model.tvec(dD_total)
        g = []
        for b, d in zip(blk, [dD] if restricted else [dD[0], dD[1]]):
            gh = np.einsum("Pvo,P->vo", b["Bvo"], u0)
            A = np.einsum("mv,Pmn->Pvn", b["Cv"], T @ d, optimize=True)
            gx = np.einsum("Pvn,Pno->vo", A, np.einsum("PQ,Qmo->Pmo", W, b["TCo"]), optimize=True)
            g.append(gh - (hyb / (2.0 if restricted else 1.0)) * gx)
        return g

    return kvec, rhs, split, np.concatenate([b["om"].ravel() for b in blk])


def correct_hx(model, mo_e, mo_c, mo_occ, dD, hyb, omega_floor=1e-3, max_rot=0.3, rtol=1e-6, maxiter=200):
    """Hartree plus exact-exchange model: K~ = K~_H - hyb K~_x, both resolved in the model's auxiliary basis,
    (mn|ls) ~ sum_PQ (mn|P) W_PQ (Q|ls), as for the exchange integrals in TDDFT-ris.  The system is solved by
    MINRES with Omega as preconditioner (it need not be positive).  Needs a dense-store model."""
    restricted = np.ndim(mo_c) == 2
    chans = _channels(mo_e, mo_c, mo_occ)
    kvec, rhs, split, om = hx_operator(model, chans, restricted, hyb, omega_floor)
    b_rhs = -np.concatenate([x.ravel() for x in rhs(dD)])
    n = om.size
    count = [0]

    def matvec(x):
        count[0] += 1
        return om * x + np.concatenate([y.ravel() for y in kvec(split(x))])

    op = LinearOperator((n, n), matvec=matvec, dtype=float)
    prec = LinearOperator((n, n), matvec=lambda y: y / om, dtype=float)
    x, _ = minres(op, b_rhs, M=prec, maxiter=maxiter, rtol=rtol)
    kappas, kmax = _cap(split(x), max_rot)
    if restricted:
        return _rotate(mo_c, chans[0][4], chans[0][5], kappas[0]), kmax, count[0]
    return np.array([_rotate(mo_c[s], chans[s][4], chans[s][5], kappas[s]) for s in range(2)]), kmax, count[0]


def correct_fock(model, mo_e, mo_c, mo_occ, fprime, dD, omega_floor=1e-3, max_rot=0.3, intraband=True):
    """The model correction applied to the Fock matrix, valid for fractional (smeared) occupations.

    The model density solves D* = Aufbau(F_bar + K~ (D* - D_bar)); linearizing Aufbau around F_bar gives the
    Fock correction u = T^T W y with (1 - chi~ W) y = T (D_R - D_bar), where chi~ = T chi0 T^T is the
    independent-particle response at fixed N projected on the auxiliary basis: in the MO basis of F_bar,
    chi0[dF]_pq = L_pq dF_pq - delta_pq f'_p dmu, L_pq = (f_p - f_q) / (e_p - e_q) (f'_p when degenerate),
    dmu = sum_p f'_p dF_pp / sum_p f'_p.  The f' (intraband) terms are the density of states at the Fermi level,
    the slope of the cumulative LDOS; intraband=False drops them.  With integer occupations this is the low-rank
    correction to first order.  Returns the AO Fock correction (per spin for unrestricted orbitals), scaled so
    that the implied first-order rotation of pairs with different occupation stays below max_rot, and that
    rotation before scaling."""
    restricted = np.ndim(mo_c) == 2
    chans = [(mo_e, mo_c, mo_occ, fprime)] if restricted else \
        [(mo_e[s], mo_c[s], mo_occ[s], fprime[s]) for s in range(2)]
    dD_total = dD if restricted else dD[0] + dD[1]
    t0 = model.tvec(dD_total)
    naux = t0.size
    chi = np.zeros((naux, naux))
    tmo = []
    for e, c, f, fp in chans:
        tm = np.einsum("mp,Pmn,nq->Ppq", c, model.T, c, optimize=True)
        de = e[:, None] - e[None, :]
        df = f[:, None] - f[None, :]
        if np.any(fp != 0):
            far = np.abs(de) > 1e-8
            L = np.where(far, df / np.where(far, de, 1.0), 0.5 * (fp[:, None] + fp[None, :]))
            if not intraband:
                np.fill_diagonal(L, 0.0)
        else:
            L = df / (np.where(de >= 0, 1.0, -1.0) * np.maximum(np.abs(de), omega_floor))
        tf = tm.reshape(naux, -1)
        chi += (tf * L.ravel()) @ tf.T
        if intraband and fp.sum() < -1e-12:
            a = np.einsum("p,Ppp->P", fp, tm)
            chi -= np.outer(a, a) / fp.sum()
        tmo.append((tm, de, df))
    y = np.linalg.solve(np.eye(naux) - chi @ model.W, t0)
    u = np.einsum("Pmn,P->mn", model.T, model.W @ y)
    rot = 0.0
    for (e, c, f, fp), (tm, de, df) in zip(chans, tmo):
        umo = c.T @ u @ c
        mask = np.abs(df) > 1e-3 * (2.0 if restricted else 1.0)
        if mask.any():
            rot = max(rot, (np.abs(umo[mask]) / np.maximum(np.abs(de[mask]), omega_floor)).max())
    scale = min(1.0, max_rot / rot) if rot > 0 else 1.0
    return (scale * u if restricted else np.array([scale * u, scale * u])), rot


def correct_exact(mf, mo_e, mo_c, mo_occ, dD, omega_floor=1e-3, max_rot=0.3, maxiter=30, rtol=1e-3):
    """The same correction with the exact response kernel; each product costs a response (J/K) build.
    Returns (mo_coeff, max|kappa| before the cap, number of response builds)."""
    restricted = np.ndim(mo_c) == 2
    chans = _channels(mo_e, mo_c, mo_occ)
    vind = mf.gen_response(mo_c, mo_occ, hermi=1)
    count = [0]

    def resp(dm):
        count[0] += 1
        return vind(dm)

    def vo_of(v):
        return [Cv.T @ (v if restricted else v[s]) @ Co for s, (eo, ev, Co, Cv, o, vv) in enumerate(chans)]

    shapes = [(len(c[5]), len(c[4])) for c in chans]
    Dg = np.concatenate([np.maximum(c[1][:, None] - c[0][None, :], omega_floor).ravel() for c in chans])
    g = np.concatenate([x.ravel() for x in vo_of(resp(dD))])

    def split(x):
        out, i0 = [], 0
        for nv, no in shapes:
            out.append(x[i0:i0 + nv * no].reshape(nv, no))
            i0 += nv * no
        return out

    def dm_of(ks):
        if restricted:
            eo, ev, Co, Cv, o, v = chans[0]
            d = Cv @ ks[0] @ Co.T * 2.0
            return d + d.T
        dm = []
        for (eo, ev, Co, Cv, o, v), k in zip(chans, ks):
            d = Cv @ k @ Co.T
            dm.append(d + d.T)
        return np.array(dm)

    def matvec(x):
        return Dg * x + np.concatenate([y.ravel() for y in vo_of(resp(dm_of(split(x))))])

    n = Dg.size
    H = LinearOperator((n, n), matvec=matvec, dtype=float)
    P = LinearOperator((n, n), matvec=lambda y: y / Dg, dtype=float)
    x, _ = minres(H, -g, M=P, maxiter=maxiter, rtol=rtol)
    kappas, kmax = _cap(split(x), max_rot)
    if restricted:
        return _rotate(mo_c, chans[0][4], chans[0][5], kappas[0]), kmax, count[0]
    return np.array([_rotate(mo_c[s], chans[s][4], chans[s][5], kappas[s]) for s in range(2)]), kmax, count[0]
