"""Is a converged SCF solution a minimum or a saddle point?  Lowest eigenvalues of the internal orbital Hessian.

    python classify.py --system tmqm:2:triplet --method mk_spd --out x.json

Reruns the SCF exactly as bench.py does (same guess and settings, so the same solution), then runs block Davidson
on PySCF's internal orbital Hessian for the 3 lowest roots to 1e-6, starting from the 6 lowest diagonal elements
plus 2 random vectors.  PySCF's default check (1 root, tol 1e-3, 1 start vector) is run afterwards on the same
orbitals for comparison.  Every Hessian product is one response (J/K) build.
"""
import argparse
import json
import os
import time

import numpy as np
from pyscf import lib
from pyscf.scf import stability

import bench
import mkdiis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--method", required=True, help="diis, mk_s, mk_spd or mk_jfit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    lib.num_threads(args.threads)
    mol, xc, guess = bench.make_system(args.system)
    mf = bench.make_mf(mol, xc)
    dm0 = mf.get_init_guess(key=guess)
    model = {"diis": None, "mk_spd": mkdiis.AuxHartreeModel("spd"), "mk_s": mkdiis.AuxHartreeModel("s"),
             "mk_jfit": mkdiis.AuxHartreeModel("def2-universal-jfit")}[args.method]
    res = mkdiis.run_scf(mf, dm0, model=model)
    mo_c, mo_occ = res["mo_coeff"], np.asarray(res["mo_occ"])
    rec = dict(system=args.system, method=args.method, converged=res["converged"], e_tot=res["e_tot"],
               cycles=res["cycles"])
    t = time.perf_counter()
    e, _, conv, nprod, gnorm = mkdiis.lowest_roots(mf, mo_c, mo_occ, nroots=3, tol=1e-6, max_space=60)
    rec.update(eig=e.tolist(), eig_converged=bool(np.all(conv)), nprod=nprod, grad_norm=gnorm,
               t_davidson=time.perf_counter() - t, saddle=bool(e.min() < -1e-4))

    fock = mf.get_fock(dm=mf.make_rdm1(mo_c, mo_occ))
    mf.mo_energy = (np.einsum("pi,pq,qi->i", mo_c, fock, mo_c) if np.ndim(mo_c) == 2 else
                    np.array([np.einsum("pi,pq,qi->i", mo_c[k], fock[k], mo_c[k]) for k in range(2)]))
    mf.mo_coeff, mf.mo_occ = mo_c, mo_occ
    njk = [0]
    vind0 = mf.gen_response

    def counted_gen_response(*a, **k):
        v = vind0(*a, **k)

        def w(*b, **c):
            njk[0] += 1
            return v(*b, **c)
        return w
    mf.gen_response = counted_gen_response
    fn = stability.rhf_internal if np.ndim(mo_c) == 2 else stability.uhf_internal
    _, stable = fn(mf, nroots=1, tol=1e-3, verbose=0, return_status=True)
    rec.update(pyscf_default_stable=bool(stable), pyscf_default_nresp=njk[0])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".part", "w") as f:
        json.dump(rec, f)
    os.replace(args.out + ".part", args.out)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
