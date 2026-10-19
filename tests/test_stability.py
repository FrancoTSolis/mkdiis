import numpy as np

from mkdiis import internal_stability, lowest_roots


def test_block_davidson_finds_a_stable_closed_shell(gly_rks):
    mf = gly_rks
    e, vecs, conv, nprod, gnorm = lowest_roots(mf, mf.mo_coeff, mf.mo_occ, nroots=3, tol=1e-6, max_space=60)
    assert np.all(conv) and e.min() > 0 and len(vecs) == 3 and nprod > 0 and gnorm < 1e-3
    mo, stable = internal_stability(mf, mf.mo_coeff, mf.mo_occ)
    assert stable and np.allclose(mo, mf.mo_coeff)


def test_block_and_pyscf_checks_agree_on_a_triplet(gly_uks):
    mf = gly_uks
    _, stable_block = internal_stability(mf, mf.mo_coeff, mf.mo_occ, mode="block")
    _, stable_pyscf = internal_stability(mf, mf.mo_coeff, mf.mo_occ, mode="pyscf")
    assert stable_block == stable_pyscf
