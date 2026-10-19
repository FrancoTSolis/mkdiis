"""MK-DIIS against DIIS on hexacene (PBE/def2-SVP): long, small-gap molecules are where DIIS sloshes most.

    python quickstart.py

model=None runs this package's DIIS without the correction, which reproduces PySCF's DIIS cycle by cycle.
"""
import numpy as np
from pyscf import dft, gto

import mkdiis


def acene(n, a=1.40, ch=1.09):
    """Linear acene with n rings (C-C 1.40, C-H 1.09 Angstrom)."""
    atoms, h = [], np.sqrt(3) / 2 * a
    xs = [i * 2 * h for i in range(n + 1)]
    for x in xs:
        atoms += [("C", x, a / 2), ("C", x, -a / 2)]
    for i in range(n):
        xm = xs[i] + h
        atoms += [("C", xm, a), ("C", xm, -a), ("H", xm, a + ch), ("H", xm, -a - ch)]
    for x, sgn in [(xs[0], -1), (xs[-1], 1)]:
        for y in (a / 2, -a / 2):
            atoms.append(("H", x + sgn * ch * np.sqrt(3) / 2, y + np.sign(y) * ch / 2))
    return [(s, x, y, 0.0) for s, x, y in atoms]


mol = gto.M(atom=acene(6), basis="def2-svp", verbose=0)
print(f"hexacene, {mol.natm} atoms, {mol.nao} basis functions, PBE/def2-SVP")
for label, model in (("DIIS", None), ("MK-DIIS", "spd")):
    mf = dft.RKS(mol, xc="pbe").density_fit()
    e = mkdiis.kernel(mf, model=model)
    st = mf.mkdiis_stats
    print(f"{label:8s} converged {mf.converged}   E = {e:.10f}   Fock builds {st['nfock']:3d}   "
          f"model {100 * st['t_model'] / st['wall']:.1f}% of {st['wall']:.1f} s")
