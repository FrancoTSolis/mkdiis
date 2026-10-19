"""Record every iterate of DIIS and MK-DIIS on a bench.py system, for the real-molecule sloshing figure.

    python real_trace.py acene:8            # writes data/trace_acene8.npz

The density of every Fock build is captured by wrapping mf.get_veff, so mkdiis.run_scf runs unchanged
(DIIS: no model; MK-DIIS: MK(spd) with the AO-direct CG solver, as for the large molecules in the paper).
Per iterate it stores Mulliken atomic charges and the orbital-gradient norm.  The reference charges come
from an extra MK-DIIS run converged to 1e-12 Eh.

The model must be kept in double precision: the minimal aux basis has a Coulomb metric with cond(J) ~ 1e10 on
acene-8, and in single precision J^-1 amplifies the rounding of the fitted density to ~1% of the model potential,
so that the first corrected step breaks the inversion symmetry of the acene by 0.06 e.

    python real_trace.py acene:8 --methods mk     # redo only MK-DIIS, keep the DIIS arrays of the file
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

import bench  # noqa: E402
import mkdiis  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def trace(spec, method, conv_tol=1e-9, max_memory=16000):
    mol, xc, guess = bench.make_system(spec)
    mf = bench.make_mf(mol, xc, max_memory)
    dm0 = mf.get_init_guess(key=guess)
    dms, veff = [], mf.get_veff

    def recording_veff(*args, **kwargs):
        dms.append(np.array(args[1] if len(args) > 1 else kwargs["dm"]))
        return veff(*args, **kwargs)

    mf.get_veff = recording_veff
    model = None if method == "diis" else mkdiis.AuxHartreeModel("spd", store="tril", screen=1e-10)
    t0 = time.perf_counter()
    res = mkdiis.run_scf(mf, dm0, model=model, conv_tol=conv_tol, solver="cg")
    s = mol.intor("int1e_ovlp")
    ao_atom = np.zeros(mol.nao, int)
    for i, (_, _, a0, a1) in enumerate(mol.aoslice_by_atom()):
        ao_atom[a0:a1] = i
    pops = np.array([np.bincount(ao_atom, weights=np.einsum("ij,ji->i", d, s), minlength=mol.natm) for d in dms])
    z = mol.atom_charges()
    print(f"{spec} {method}: converged {res['converged']}, {res['nfock']} Fock builds, E = {res['e_tot']:.10f}, "
          f"{time.perf_counter() - t0:.0f} s", flush=True)
    return dict(charges=z[None, :] - pops, grad=np.array([h[2] for h in res["hist"]]), e=res["e_tot"],
                nfock=res["nfock"], converged=res["converged"]), mol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    ap.add_argument("--out", default=None)
    ap.add_argument("--methods", default="diis,mk")
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, "data", "trace_" + args.system.replace(":", "") + ".npz")
    old = dict(np.load(out)) if os.path.exists(out) else {}
    runs = {}
    for method in args.methods.split(","):
        runs[method], mol = trace(args.system, method)
    if "ref" not in old:
        old["ref"] = trace(args.system, "mk", conv_tol=1e-12)[0]["charges"][-1]
    old.update(system=args.system, coords=mol.atom_coords(unit="Angstrom"),
               symbols=np.array([mol.atom_symbol(i) for i in range(mol.natm)]),
               **{f"{m}_{k}": v for m, r in runs.items() for k, v in r.items()})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, **old)
    print("wrote", out)


if __name__ == "__main__":
    main()
