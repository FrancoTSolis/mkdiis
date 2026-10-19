import numpy as np
from conftest import sym
from pyscf.soscf import newton_ah

from mkdiis import AuxHartreeModel
from mkdiis.correction import _channels, _rotate, _rotate_svd, correct_cg, correct_lowrank


def test_woodbury_solve_equals_dense_solve(gly, gly_rks, rng):
    model = AuxHartreeModel("spd").build(gly)
    mo_e, mo_c, mo_occ = gly_rks.mo_energy, gly_rks.mo_coeff, gly_rks.mo_occ
    (eo, ev, Co, Cv, o, v), = _channels(mo_e, mo_c, mo_occ)
    B = model.vo(Co, Cv)
    K = 4.0 * B.reshape(model.naux, -1).T @ model.W @ B.reshape(model.naux, -1)
    Om = np.maximum(ev[:, None] - eo[None, :], 1e-3).ravel()
    dD = sym(rng, gly.nao)
    g = np.einsum("Pvo,P->vo", B, model.W @ model.tvec(dD)).ravel()
    x_dense = np.linalg.solve(np.diag(Om) + K, -g)
    c_new, _, _ = correct_lowrank(model, mo_e, mo_c, mo_occ, dD, max_rot=10.0)
    kappa = Cv.T @ gly.intor("int1e_ovlp") @ c_new[:, o]
    assert np.linalg.norm(kappa.ravel() - x_dense) / np.linalg.norm(x_dense) < 5e-2


def test_cg_solve_equals_woodbury(gly, gly_rks, gly_triplet, gly_uks, rng):
    d = sym(rng, gly.nao)
    for mol, mf, dd in ((gly, gly_rks, d), (gly_triplet, gly_uks, np.array([d, 0.3 * d]))):
        dense = AuxHartreeModel("spd").build(mol)
        tril = AuxHartreeModel("spd", store="tril").build(mol)
        ca, _, _ = correct_lowrank(dense, mf.mo_energy, mf.mo_coeff, mf.mo_occ, dd, max_rot=10.0)
        cb, _, _ = correct_cg(tril, mf.mo_energy, mf.mo_coeff, mf.mo_occ, dd, max_rot=10.0, rtol=1e-10, maxiter=500)
        assert np.abs(np.asarray(ca) - np.asarray(cb)).max() < 1e-7


def test_svd_rotation_equals_expm(gly_rks, rng):
    (eo, ev, Co, Cv, o, v), = _channels(gly_rks.mo_energy, gly_rks.mo_coeff, gly_rks.mo_occ)
    kap = rng.standard_normal((len(v), len(o))) * 0.1
    c1, c2 = _rotate(gly_rks.mo_coeff, o, v, kap), _rotate_svd(gly_rks.mo_coeff, o, v, kap)
    assert np.abs(c1 - c2).max() < 1e-12


def test_exact_kernel_matches_pyscf_orbital_hessian(gly_rks, rng):
    mf = gly_rks
    (eo, ev, Co, Cv, o, v), = _channels(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    _, h_op, _ = newton_ah.gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ)
    vind = mf.gen_response(mf.mo_coeff, mf.mo_occ, hermi=1)
    x = rng.standard_normal((len(v), len(o))) * 1e-3
    dm1 = Cv @ x @ Co.T * 2.0
    mine = (ev[:, None] - eo[None, :]) * x + Cv.T @ vind(dm1 + dm1.T) @ Co
    ref = h_op(x.ravel())
    assert np.linalg.norm(2 * mine.ravel() - ref) / np.linalg.norm(ref) < 1e-5
