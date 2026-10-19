"""The SCF driver: PySCF's DIIS schedule with an optional model-kernel correction of every Roothaan step."""
import time

import numpy as np

from .correction import correct_cg, correct_exact, correct_fock, correct_hx, correct_lowrank, fermi_occ
from .diis import ADIIS, CDIIS
from .model import AuxHartreeModel

PRESETS = {"spd": "spd", "s": "s", "jfit": "def2-universal-jfit"}


def _orth(s, lindep=1e-8):
    w, v = np.linalg.eigh(s)
    keep = w > lindep
    return v[:, keep] / np.sqrt(w[keep])


def _commutator(fock, dm, s1e, X):
    if np.ndim(fock) == 2:
        return X.T @ (fock @ dm @ s1e - s1e @ dm @ fock) @ X
    return np.array([X.T @ (fock[s] @ dm[s] @ s1e - s1e @ dm[s] @ fock[s]) @ X for s in range(2)])


def run_scf(mf, dm0, model=None, max_cycle=150, conv_tol=1e-9, conv_tol_grad=None, diis_space=8,
            diis_start_cycle=1, omega_floor=1e-3, max_rot=0.3, accel="cdiis", adiis_switch=1e-3,
            adiis_max_cycle=30, model_off_grad=0.0, model_on_grad=np.inf, solver="lowrank", cg_rtol=1e-3,
            cg_maxiter=50, smearing=0.0, model_form="rotate", intraband=True, log=None):
    """Converge a PySCF RHF/RKS or UHF/UKS object with DIIS and an optional model-kernel correction.

    mf, dm0: the PySCF mean-field object and the initial density matrix (mf.get_init_guess()).
    model: None (plain DIIS), an AuxHartreeModel (built here), or "exact" (the exact-kernel bound).
    solver: "lowrank" (Woodbury, dense-store model) or "cg" (AO-direct conjugate gradients, any store).
    accel: "cdiis", or "adiis" (ADIIS until the DIIS error norm drops below adiis_switch or after adiis_max_cycle
        cycles, then CDIIS, as in Q-Chem's ADIIS_DIIS).
    model_off_grad: apply the correction only while the orbital-gradient norm of the previous cycle is above this
        value (0 = always).  model_on_grad: ... and only once it has dropped below this value (inf = from the first
        cycle), so that plain DIIS takes the first steps; 0.05 is the delayed start used for open shells.
    smearing: Fermi-Dirac width in Hartree (0 = aufbau).  With smearing the correction is applied to the Fock
        matrix (correct_fock, dense-store model), convergence uses the free energy E - sigma S and the norm of the
        commutator / sqrt(2) in place of the orbital gradient.  model_form="fock" uses that form without smearing.
    intraband: keep the f' (density of states at the Fermi level) terms of the smeared response.

    Convergence and the DIIS schedule follow PySCF: no extrapolation before diis_start_cycle, |dE| < conv_tol and
    orbital-gradient norm < conv_tol_grad (default sqrt(conv_tol)).  A Fock build is counted for the initial guess
    and for every cycle.  Returns a dict with the statistics (converged, e_tot, cycles, nfock, nresp, wall, t_model,
    t_fock, hist = [(cycle, energy, gradient norm, max|kappa|)]) and the final dm, fock, mo_coeff and mo_occ.
    """
    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
    mol = mf.mol
    t0 = time.perf_counter()
    s1e, h1e = mf.get_ovlp(), mf.get_hcore()
    X = _orth(s1e)
    hyb = 1.0
    if model is not None and model != "exact":
        model.build(mol)
        if getattr(model, "exchange", False) and hasattr(mf, "xc"):
            hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)[2]
    t_model = t_model_build = time.perf_counter() - t0

    dm = dm0
    restricted = np.ndim(dm0) == 2
    tf = time.perf_counter()
    vhf = mf.get_veff(mol, dm)
    t_fock = time.perf_counter() - tf
    e_tot = mf.energy_tot(dm, h1e, vhf)
    e_obj, entropy = e_tot, 0.0
    nfock, nresp, converged = 1, 0, False
    diis = CDIIS(diis_space)
    adiis = ADIIS(diis_space) if accel == "adiis" else None
    hist, napply = [], []
    fock_new = mf.get_fock(h1e, s1e, vhf, dm)
    mo_c, mo_occ = None, None
    for cycle in range(max_cycle):
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        if np.ndim(fock) == 2:
            err = X.T @ (fock @ dm @ s1e - s1e @ dm @ fock) @ X
        else:
            err = np.array([X.T @ (fock[s] @ dm[s] @ s1e - s1e @ dm[s] @ fock[s]) @ X for s in range(2)])
        if cycle >= diis_start_cycle:
            F_bar, D_bar = diis.update(fock, dm, err)
            if adiis is not None:
                F_ad, D_ad = adiis.update(fock, dm)
                if np.linalg.norm(err) > adiis_switch and cycle < adiis_max_cycle:
                    F_bar, D_bar = F_ad, D_ad
                else:
                    adiis = None
        else:
            F_bar, D_bar = fock, dm
        mo_e, mo_c = mf.eig(F_bar, s1e)
        if smearing > 0:
            mo_occ, entropy, fprime = fermi_occ(mo_e, mol.nelec, smearing, restricted)
        else:
            mo_occ, fprime = mf.get_occ(mo_e, mo_c), None
        kmax = 0.0
        gprev = hist[-1][2] if hist else np.inf
        if model is not None and gprev > model_off_grad and (gprev < model_on_grad or model_on_grad == np.inf):
            tm = time.perf_counter()
            dD = mf.make_rdm1(mo_c, mo_occ) - D_bar
            if smearing > 0 or model_form == "fock":
                if fprime is None:
                    fprime = np.zeros_like(np.asarray(mo_occ, dtype=float))
                u, kmax = correct_fock(model, mo_e, mo_c, mo_occ, fprime, dD, omega_floor, max_rot, intraband)
                mo_e, mo_c = mf.eig(F_bar + u, s1e)
                if smearing > 0:
                    mo_occ, entropy, fprime = fermi_occ(mo_e, mol.nelec, smearing, restricted)
                else:
                    mo_occ = mf.get_occ(mo_e, mo_c)
            elif model == "exact":
                mo_c, kmax, nr = correct_exact(mf, mo_e, mo_c, mo_occ, dD, omega_floor, max_rot)
                nresp += nr
            elif getattr(model, "exchange", False):
                mo_c, kmax, na = correct_hx(model, mo_e, mo_c, mo_occ, dD, hyb, omega_floor, max_rot)
                napply.append(na)
            elif solver == "cg":
                mo_c, kmax, na = correct_cg(model, mo_e, mo_c, mo_occ, dD, omega_floor, max_rot, cg_rtol, cg_maxiter)
                napply.append(na)
            else:
                mo_c, kmax, _ = correct_lowrank(model, mo_e, mo_c, mo_occ, dD, omega_floor, max_rot)
            t_model += time.perf_counter() - tm
        dm_last, vhf_last, e_last = dm, vhf, e_obj
        dm = mf.make_rdm1(mo_c, mo_occ)
        tf = time.perf_counter()
        vhf = mf.get_veff(mol, dm, dm_last, vhf_last)
        t_fock += time.perf_counter() - tf
        nfock += 1
        e_tot = mf.energy_tot(dm, h1e, vhf)
        fock_new = mf.get_fock(h1e, s1e, vhf, dm)
        if smearing > 0:
            e_obj = e_tot - smearing * entropy
            gnorm = np.linalg.norm(_commutator(fock_new, dm, s1e, X)) / np.sqrt(2.0)
        else:
            e_obj = e_tot
            gnorm = np.linalg.norm(mf.get_grad(mo_c, mo_occ, fock_new))
        hist.append((cycle, float(e_obj), float(gnorm), float(kmax)))
        if log is not None:
            log(f"cycle {cycle:3d}  E = {e_obj:.12f}  dE = {e_obj - e_last:.3e}  |g| = {gnorm:.3e}  kmax = {kmax:.3f}")
        if not np.isfinite(e_tot):
            break
        if abs(e_obj - e_last) < conv_tol and gnorm < conv_tol_grad:
            converged = True
            break
    out = dict(converged=converged, e_tot=float(e_tot), cycles=len(hist), nfock=nfock, nresp=nresp,
               wall=time.perf_counter() - t0, t_model=t_model, t_model_build=t_model_build, t_fock=t_fock,
               napply=napply, hist=hist,
               mo_occ=None if mo_occ is None else np.asarray(mo_occ).tolist(), mo_coeff=mo_c, dm=dm, fock=fock_new)
    if smearing > 0:
        out.update(e_free=float(e_obj), entropy=float(entropy), smearing=smearing)
    return out


def kernel(mf, model="spd", dm0=None, solver="auto", **kwargs):
    """Converge `mf` with MK-DIIS and store the result on it, like mf.kernel(); returns the total energy.

    model: "spd" (default), "s", "jfit", "exact", None (plain DIIS) or an AuxHartreeModel.
    solver: "auto" uses the low-rank solver up to 1000 basis functions and the AO-direct conjugate-gradient solver
        with the lower-triangle store and pair screening above, as for the large molecules of the paper.
    Other keyword arguments go to run_scf.  Afterwards mf.e_tot, mf.converged, mf.mo_energy, mf.mo_coeff and
    mf.mo_occ hold the result (canonical orbitals of the final Fock matrix) and mf.mkdiis_stats the statistics.
    """
    mol = mf.mol
    if solver == "auto":
        solver = "cg" if mol.nao > 1000 else "lowrank"
    if isinstance(model, str) and model != "exact":
        large = solver == "cg"
        model = AuxHartreeModel(PRESETS.get(model, model), store="tril" if large else "dense",
                                screen=1e-10 if large else 0.0)
    if dm0 is None:
        dm0 = mf.get_init_guess()
    res = run_scf(mf, dm0, model=model, solver=solver, **kwargs)
    mo_e, mo_c = mf.eig(res["fock"], mf.get_ovlp())
    if res.get("smearing", 0) > 0:
        mo_occ = fermi_occ(mo_e, mol.nelec, res["smearing"], np.ndim(mo_c) == 2)[0]
    else:
        mo_occ = mf.get_occ(mo_e, mo_c)
    mf.mo_energy, mf.mo_coeff, mf.mo_occ = mo_e, mo_c, mo_occ
    mf.e_tot, mf.converged = res["e_tot"], res["converged"]
    mf.mkdiis_stats = {k: v for k, v in res.items() if k not in ("dm", "fock", "mo_coeff", "mo_occ")}
    return mf.e_tot
