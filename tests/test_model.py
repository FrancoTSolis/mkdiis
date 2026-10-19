import numpy as np
from conftest import acene, sym
from pyscf import gto

from mkdiis import AuxHartreeModel
from mkdiis.correction import _channels


def test_jfit_model_reproduces_exact_coulomb(gly, gly_rks, rng):
    model = AuxHartreeModel("def2-universal-jfit").build(gly)
    d = sym(rng, gly.nao)
    vj = gly_rks.get_j(gly, d)
    assert np.linalg.norm(model.apply(d) - vj) / np.linalg.norm(vj) < 1e-2


def test_low_rank_form_equals_ao_route(gly, gly_rks, rng):
    model = AuxHartreeModel("spd").build(gly)
    (eo, ev, Co, Cv, o, v), = _channels(gly_rks.mo_energy, gly_rks.mo_coeff, gly_rks.mo_occ)
    B = model.vo(Co, Cv).reshape(model.naux, -1)
    K = 4.0 * B.T @ model.W @ B
    kap = rng.standard_normal(K.shape[0]) * 1e-3
    dm1 = Cv @ kap.reshape(len(v), len(o)) @ Co.T * 2.0
    lhs = (Cv.T @ model.apply(dm1 + dm1.T) @ Co).ravel()
    assert np.allclose(lhs, K @ kap, rtol=1e-8, atol=1e-12)


def test_lower_triangle_store_equals_dense(gly, rng):
    d = sym(rng, gly.nao)
    ref = AuxHartreeModel("spd").build(gly).apply(d)
    for kw in ({}, dict(screen=1e-10)):
        got = AuxHartreeModel("spd", store="tril", **kw).build(gly).apply(d)
        assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-8


def test_pair_screening_on_an_acene(rng):
    mol = gto.M(atom=acene(5), basis="def2-svp", verbose=0)
    d = sym(rng, mol.nao)
    full = AuxHartreeModel("spd", store="tril").build(mol)
    screened = AuxHartreeModel("spd", store="tril", screen=1e-10).build(mol)
    assert screened.npair < full.npair
    ref = full.apply(d)
    assert np.linalg.norm(screened.apply(d) - ref) / np.linalg.norm(ref) < 1e-6


def test_minimal_basis_sizes(gly):
    s = AuxHartreeModel("s").build(gly)
    spd = AuxHartreeModel("spd").build(gly)
    nheavy = sum(gly.atom_symbol(i) != "H" for i in range(gly.natm))
    assert s.naux == gly.natm
    assert spd.naux == gly.natm + nheavy * (3 + 5)
