"""MK-DIIS: model-kernel corrected DIIS for PySCF.

    from pyscf import gto, dft
    import mkdiis

    mf = dft.RKS(gto.M(atom=..., basis="def2-svp"), xc="b3lyp")
    mkdiis.kernel(mf)              # MK(spd); mf.e_tot, mf.mo_coeff, ... as after mf.kernel()
    print(mf.mkdiis_stats["nfock"])
"""
from .model import AuxHartreeModel
from .scf import kernel, run_scf
from .stability import internal_stability, lowest_roots

__version__ = "0.1.0"
__all__ = ["AuxHartreeModel", "internal_stability", "kernel", "lowest_roots", "run_scf"]
