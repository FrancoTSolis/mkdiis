"""Open shells: several SCF solutions compete, so the first steps choose the basin.

    python open_shell.py

The protocol of the paper: let DIIS take the first steps and switch the model on once the orbital gradient is below
0.05, then check the converged solution for an internal instability with block Davidson and, if it is unstable,
restart along the unstable mode.  The system is a tmQM titanium complex as a triplet (UB3LYP/def2-SVP).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks"))
import bench  # noqa: E402

import mkdiis  # noqa: E402

mol, xc, guess = bench.make_system("tmqm:0:triplet")
print(f"tmQM #0 ({mol.atom_symbol(0)}...), {mol.natm} atoms, {mol.nao} basis functions, triplet, UB3LYP/def2-SVP")
for label, model, on in (("DIIS", None, np.inf), ("MK-DIIS, model below |g| = 0.05", mkdiis.AuxHartreeModel("spd"), 0.05)):
    mf = bench.make_mf(mol, xc)
    res = mkdiis.run_scf(mf, mf.get_init_guess(key=guess), model=model, model_on_grad=on)
    builds = res["nfock"]
    for check in range(3):
        mo, stable = mkdiis.internal_stability(mf, res["mo_coeff"], np.asarray(res["mo_occ"]))
        print(f"{label}: E = {res['e_tot']:.8f} after {builds} Fock builds, {'stable' if stable else 'unstable'}")
        if stable:
            break
        res = mkdiis.run_scf(mf, mf.make_rdm1(mo, np.asarray(res["mo_occ"])), model=model, model_on_grad=on)
        builds += res["nfock"]
