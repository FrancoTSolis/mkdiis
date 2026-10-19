"""Metals: Fermi smearing, with the model applied to the Fock matrix (including the Fermi-level term).

    python metal_cluster.py

Icosahedral Al13- (PBE/def2-SVP, density-fitted), smearing width 0.005 Hartree.  With fractional occupations the
density also responds by changing the occupations near the Fermi level; that term is the density of states at
the Fermi level, and the model includes it.  Convergence is on the free energy E - sigma S.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks"))
import bench  # noqa: E402

import mkdiis  # noqa: E402

mol, xc, guess = bench.make_system("ico:Al:-1")
print(f"Al13-, {mol.nao} basis functions, PBE/def2-SVP, Fermi smearing 0.005 Hartree")
for label, model in (("smeared DIIS", None), ("smeared MK-DIIS", mkdiis.AuxHartreeModel("spd"))):
    mf = bench.make_mf(mol, xc)
    res = mkdiis.run_scf(mf, mf.get_init_guess(key=guess), model=model, smearing=0.005)
    print(f"{label:16s} converged {res['converged']}   free energy {res['e_free']:.8f}   Fock builds {res['nfock']}")
