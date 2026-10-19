"""1D toy model of charge sloshing in the SCF iteration and of the model-kernel (MK) correction.

A half-filled chain of n sites with alternating hopping (a gapped conjugated chain) in a weak electric field
along the chain.  The mean-field potential is v = K (n - 1) with K = Hartree (Coulomb between Gaussian site
densities) + a local exchange-correlation term.  The MK correction uses only the Hartree part, fitted in one s
Gaussian per site with a different exponent, so its model kernel is right at long range and wrong at short
range, like MK(spd) in the molecular code.  Every run starts from neutral sites (the analogue of a SAD guess).

The solver mirrors mkdiis.run_scf: CDIIS on the commutator from the second cycle, extrapolated Fock and density
matrices, a Roothaan step, and optionally the correction (Omega + K~) kappa = -g, g = [K~ (D_R - D_bar)]_vo,
with Omega floored and max|kappa| capped at 0.3 rad.  Energies are in eV and lengths in Angstrom.

    python toy_chain.py          # Fock builds for each solver and chain length

The default parameters put the chain in the linear-response regime in which the longest (dipole-like) charge
mode has a Roothaan multiplier well above one: the plain iteration sloshes, DIIS damps it, MK-DIIS does not
start it.  A much stronger field moves frontier levels across the gap; the occupied-virtual correction cannot
undo such occupation flips (like decacene in the paper).
"""
import warnings

import numpy as np
from scipy.special import erf

# numpy 2.x with Apple Accelerate raises spurious floating-point warnings inside matmul.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

COUL = 14.399645  # e^2 / (4 pi eps0) in eV Angstrom


def coulomb_gauss(r, a, b):
    """Coulomb energy between unit s Gaussian charges with exponents a, b (Angstrom^-2) at distance r."""
    p = np.sqrt(a * b / (a + b))
    r = np.asarray(r, float)
    out = np.full(r.shape, COUL * 2.0 * p / np.sqrt(np.pi))
    far = r > 1e-10
    out[far] = COUL * erf(p * r[far]) / r[far]
    return out


class Chain:
    def __init__(self, n=60, a=1.4, t1=2.45, t2=2.35, drop=0.5, beta=0.9, fxc=1.5, aux=0.45):
        """drop: potential-energy difference (eV) of an electron between the two ends from the external field."""
        self.n, self.nocc = n, n // 2
        self.x = np.arange(n) * a
        h = np.zeros((n, n))
        for i in range(n - 1):
            h[i, i + 1] = h[i + 1, i] = -(t1 if i % 2 == 0 else t2)
        self.field = drop * (self.x - self.x.mean()) / (self.x[-1] - self.x[0])
        self.h0 = h + np.diag(self.field)
        r = np.abs(self.x[:, None] - self.x[None, :])
        self.K_h = coulomb_gauss(r, beta, beta)
        self.K = self.K_h - fxc * np.eye(n)
        T = coulomb_gauss(r, aux, beta)
        J = coulomb_gauss(r, aux, aux)
        self.Kt = T.T @ np.linalg.solve(J, T)

    def fock(self, D):
        return self.h0 + np.diag(self.K @ (np.diag(D) - 1.0))

    def guess(self):
        return np.eye(self.n)

    def units(self, q):
        """Charge per strongly bonded pair of sites, which removes the alternation inside each pair."""
        q = np.asarray(q)
        return q.reshape(q.shape[:-1] + (-1, 2)).sum(-1)


class CDIIS:
    def __init__(self, space=8):
        self.space, self.F, self.D, self.E = space, [], [], []

    def update(self, F, D, err):
        self.F.append(F), self.D.append(D), self.E.append(err.ravel())
        if len(self.E) > self.space:
            self.F.pop(0), self.D.pop(0), self.E.pop(0)
        n = len(self.E)
        B = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(i + 1):
                B[i, j] = B[j, i] = self.E[i] @ self.E[j]
        B[n, :n] = B[:n, n] = -1.0
        rhs = np.zeros(n + 1)
        rhs[n] = -1.0
        w, v = np.linalg.eigh(B)
        keep = np.abs(w) > 1e-14 * np.abs(w).max()
        c = (v[:, keep] @ ((v[:, keep].T @ rhs) / w[keep]))[:n]
        return sum(ci * Fi for ci, Fi in zip(c, self.F)), sum(ci * Di for ci, Di in zip(c, self.D))


def _rotate(C, nocc, kappa):
    """C exp(kappa^): occupied orbitals rotate into virtual ones by kappa (nvir x nocc), via its thin SVD."""
    u, s, vt = np.linalg.svd(kappa, full_matrices=False)
    Co, Cv = C[:, :nocc], C[:, nocc:]
    CoV, CvU = Co @ vt.T, Cv @ u
    out = C.copy()
    out[:, :nocc] = Co + (CoV * (np.cos(s) - 1.0)) @ vt + (CvU * np.sin(s)) @ vt
    out[:, nocc:] = Cv + (CvU * (np.cos(s) - 1.0)) @ u.T - (CoV * np.sin(s)) @ u.T
    return out


def mk_correct(chain, eps, C, dD, omega_floor=0.03, max_rot=0.3):
    """Solve (Omega + K~) kappa = -[K~ dD]_vo with the push-through (Woodbury) identity in site space."""
    no = chain.nocc
    Co, Cv = C[:, :no], C[:, no:]
    om = np.maximum(eps[no:, None] - eps[None, :no], omega_floor)          # (nvir, nocc)
    P = (Cv[:, :, None] * Co[:, None, :]).reshape(chain.n, -1)              # site j, pair (a, i)
    M = 4.0 * chain.Kt                                                     # dn = 4 P kappa
    g = P.T @ (chain.Kt @ np.diag(dD))
    x = -g / om.ravel()
    chi = (P / om.ravel()) @ P.T
    w = np.linalg.solve(np.eye(chain.n) + M @ chi, M @ (P @ x))
    kappa = (x - (P.T @ w) / om.ravel()).reshape(om.shape)
    kmax = np.abs(kappa).max()
    if kmax > max_rot:
        kappa *= max_rot / kmax
    return _rotate(C, no, kappa), kmax


def run(chain, diis=True, model=False, max_cycle=60, tol=1e-5, space=8):
    """Returns the site charges n_k - 1 of every iterate (k = 0 is the guess) and the commutator norms.
    The number of Fock builds is len(comm): one for the guess and one per step."""
    D = chain.guess()
    F = chain.fock(D)
    acc = CDIIS(space)
    charges, comm, kmaxs = [np.diag(D) - 1.0], [], []
    for cycle in range(max_cycle):
        err = F @ D - D @ F
        comm.append(np.linalg.norm(err))
        if comm[-1] < tol and cycle > 0:
            break
        if diis and cycle >= 1:
            Fb, Db = acc.update(F, D, err)
        else:
            Fb, Db = F, D
        eps, C = np.linalg.eigh(Fb)
        kmax = 0.0
        if model:
            DR = 2.0 * C[:, :chain.nocc] @ C[:, :chain.nocc].T
            C, kmax = mk_correct(chain, eps, C, DR - Db)
        kmaxs.append(kmax)
        D = 2.0 * C[:, :chain.nocc] @ C[:, :chain.nocc].T
        F = chain.fock(D)
        charges.append(np.diag(D) - 1.0)
    return dict(charges=np.array(charges), comm=np.array(comm), kmax=np.array(kmaxs),
                converged=comm[-1] < tol)


SOLVERS = {"Roothaan": dict(diis=False, model=False), "MK step": dict(diis=False, model=True),
           "DIIS": dict(diis=True, model=False), "MK-DIIS": dict(diis=True, model=True)}


if __name__ == "__main__":
    for n in (20, 40, 60, 80, 100, 120):
        ch = Chain(n=n)
        out = {name: run(ch, max_cycle=80, **kw) for name, kw in SOLVERS.items()}
        ref = out["MK-DIIS"]["charges"][-1]
        line = []
        for name, r in out.items():
            dev = np.abs(r["charges"][-1] - ref).max()
            line.append(f"{name} {len(r['comm']):3d}{'' if r['converged'] else '(nc)'} dev {dev:.0e}")
        print(f"n={n:3d}  " + "  ".join(line))
