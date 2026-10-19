"""Internal stability check of a converged solution by block Davidson on the orbital Hessian.

PySCF's default check starts from a single vector and can miss a negative root; in our tests it reported the same
solution as stable in one run and unstable in another.  Starting from the six lowest diagonal elements and two
random vectors is both cheaper and more reliable.  Every Hessian product costs one response (J/K) build.
"""
import numpy as np
from pyscf import lib
from pyscf.scf import stability as pyscf_stability
from pyscf.soscf import newton_ah


def lowest_roots(mf, mo_c, mo_occ, nroots=1, tol=1e-4, max_space=40, nstart=6, nrand=2, seed=0):
    """Lowest eigenvalues of the internal orbital Hessian, in the units of PySCF's h_op.

    Block Davidson from the `nstart` lowest diagonal elements plus `nrand` random vectors.  Returns (eigenvalues,
    eigenvectors, convergence flags, number of Hessian products, norm of the orbital gradient)."""
    unrestricted = np.ndim(mo_c) == 3
    g, h_op, h_diag = (newton_ah.gen_g_hop_uhf if unrestricted else newton_ah.gen_g_hop_rhf)(mf, mo_c, mo_occ)
    count = [0]

    def aop(xs):
        count[0] += len(xs)
        return [h_op(x).real for x in xs]

    def precond(dx, e, x0):
        hd = h_diag - e
        hd[abs(hd) < 1e-8] = 1e-8
        return dx / hd

    n = h_diag.size
    x0 = [np.eye(1, n, i).ravel() for i in np.argsort(h_diag)[:nstart]]
    rng = np.random.default_rng(seed)
    x0 += [v / np.linalg.norm(v) for v in rng.standard_normal((nrand, n))]
    conv, e, x = lib.davidson1(aop, x0, precond, tol=tol, max_cycle=200, max_space=max_space, nroots=nroots)
    e, conv = np.atleast_1d(np.asarray(e)), np.atleast_1d(np.asarray(conv))
    if isinstance(x, (list, tuple)):
        vecs = [np.asarray(v).ravel() for v in x]
    else:
        vecs = list(np.atleast_2d(np.asarray(x)))
    return e, vecs, conv, count[0], float(np.linalg.norm(g))


def internal_stability(mf, mo_c, mo_occ, mode="block", threshold=-5e-6):
    """Check a converged solution for an internal instability; returns (mo_coeff, stable).

    mode = "block": lowest root by block Davidson to 1e-4; unstable if it is below `threshold` (PySCF's -1e-5 in
                    its doubled Hessian scale).  On instability the orbitals are rotated along the root, as PySCF
                    does, so that restarting the SCF from them leaves the saddle point.
    mode = "pyscf": PySCF's own check (one start vector, one root, tol 1e-3).
    """
    fock = mf.get_fock(dm=mf.make_rdm1(mo_c, mo_occ))
    if np.ndim(mo_c) == 2:
        mf.mo_energy = np.einsum("pi,pq,qi->i", mo_c, fock, mo_c)
    else:
        mf.mo_energy = np.array([np.einsum("pi,pq,qi->i", mo_c[k], fock[k], mo_c[k]) for k in range(2)])
    mf.mo_coeff, mf.mo_occ = mo_c, mo_occ
    if mode == "pyscf":
        fn = pyscf_stability.rhf_internal if np.ndim(mo_c) == 2 else pyscf_stability.uhf_internal
        mo_new, stable = fn(mf, nroots=1, tol=1e-3, verbose=0, return_status=True)
        return np.asarray(mo_new), stable
    e, vecs, _, _, _ = lowest_roots(mf, mo_c, mo_occ)
    if e[0] > threshold:
        return np.asarray(mo_c), True
    v = vecs[0]
    if np.ndim(mo_c) == 2:
        return np.asarray(pyscf_stability._rotate_mo(mo_c, mo_occ, v)), False
    na = np.count_nonzero(mo_occ[0] > 0) * np.count_nonzero(mo_occ[0] == 0)
    return np.array([pyscf_stability._rotate_mo(mo_c[0], mo_occ[0], v[:na]),
                     pyscf_stability._rotate_mo(mo_c[1], mo_occ[1], v[na:])]), False
