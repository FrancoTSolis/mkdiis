import numpy as np
from conftest import sym
from pyscf import dft, scf

import mkdiis
from mkdiis import AuxHartreeModel, run_scf
from mkdiis.correction import correct_fock, correct_lowrank, fermi_occ


def test_plain_diis_reproduces_pyscf(gly, gly_rks):
    mf = dft.RKS(gly, xc="b3lyp")
    res = run_scf(mf, mf.get_init_guess(), model=None)
    assert res["converged"]
    assert abs(res["e_tot"] - gly_rks.e_tot) < 1e-7


def test_models_reach_the_same_solution_rks(gly, gly_rks):
    dm0 = dft.RKS(gly, xc="b3lyp").get_init_guess()
    for model in (AuxHartreeModel("s"), AuxHartreeModel("spd"), AuxHartreeModel("def2-universal-jfit"), "exact"):
        res = run_scf(dft.RKS(gly, xc="b3lyp"), dm0, model=model)
        assert res["converged"] and abs(res["e_tot"] - gly_rks.e_tot) < 1e-7


def test_models_reach_the_same_solution_uks(gly_triplet, gly_uks):
    dm0 = dft.UKS(gly_triplet, xc="b3lyp").get_init_guess()
    for model in (AuxHartreeModel("spd"), "exact"):
        res = run_scf(dft.UKS(gly_triplet, xc="b3lyp"), dm0, model=model)
        assert res["converged"] and abs(res["e_tot"] - gly_uks.e_tot) < 1e-6


def test_cg_solver_takes_the_same_steps_as_woodbury(gly):
    dm0 = dft.RKS(gly, xc="b3lyp").get_init_guess()
    a = run_scf(dft.RKS(gly, xc="b3lyp"), dm0, model=AuxHartreeModel("spd"))
    b = run_scf(dft.RKS(gly, xc="b3lyp"), dm0, model=AuxHartreeModel("spd", store="tril", screen=1e-10), solver="cg")
    assert b["converged"] and b["cycles"] == a["cycles"] and abs(b["e_tot"] - a["e_tot"]) < 1e-7


def test_adiis_with_the_model(gly, gly_rks):
    mf = dft.RKS(gly, xc="b3lyp")
    res = run_scf(mf, mf.get_init_guess(key="1e"), model=AuxHartreeModel("spd"), accel="adiis")
    assert res["converged"] and abs(res["e_tot"] - gly_rks.e_tot) < 1e-7


def test_delayed_start_of_the_model(gly_triplet, gly_uks):
    mf = dft.UKS(gly_triplet, xc="b3lyp")
    res = run_scf(mf, mf.get_init_guess(), model=AuxHartreeModel("spd"), model_on_grad=0.05)
    assert res["converged"] and abs(res["e_tot"] - gly_uks.e_tot) < 1e-6
    assert res["hist"][0][3] == 0.0


def test_kernel_stores_the_result_on_mf(gly, gly_rks):
    mf_diis = dft.RKS(gly, xc="b3lyp")
    mkdiis.kernel(mf_diis, model=None)
    mf = dft.RKS(gly, xc="b3lyp")
    e = mkdiis.kernel(mf)
    assert mf.converged and abs(e - gly_rks.e_tot) < 1e-7
    assert np.allclose(mf.mo_energy[mf.mo_occ > 0], gly_rks.mo_energy[gly_rks.mo_occ > 0], atol=1e-5)
    assert mf.mkdiis_stats["nfock"] < mf_diis.mkdiis_stats["nfock"]


def test_fermi_occupations_match_pyscf_smearing(gly):
    sigma = 0.01
    ref = scf.addons.smearing_(dft.RKS(gly, xc="b3lyp"), sigma=sigma, method="fermi")
    ref.conv_tol = 1e-11
    ref.kernel()
    occ, _, _ = fermi_occ(ref.mo_energy, gly.nelec, sigma, True)
    assert np.abs(occ - ref.mo_occ).max() < 1e-8 and abs(occ.sum() - gly.nelectron) < 1e-10
    dm0 = dft.RKS(gly, xc="b3lyp").get_init_guess()
    a = run_scf(dft.RKS(gly, xc="b3lyp"), dm0, smearing=sigma)
    assert a["converged"] and abs(a["e_tot"] - ref.e_tot) < 1e-7
    b = run_scf(dft.RKS(gly, xc="b3lyp"), dm0, model=AuxHartreeModel("spd"), smearing=sigma)
    assert b["converged"] and abs(b["e_free"] - a["e_free"]) < 1e-7


def test_fock_form_equals_rotation_form(gly, gly_rks, rng):
    mf = gly_rks
    model = AuxHartreeModel("spd").build(gly)
    d = sym(rng, gly.nao, 1e-5)
    c_rot, _, _ = correct_lowrank(model, mf.mo_energy, mf.mo_coeff, mf.mo_occ, d, max_rot=10.0)
    D_R = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    dD_rot = mf.make_rdm1(c_rot, mf.mo_occ) - D_R
    s1e = mf.get_ovlp()
    fock = s1e @ mf.mo_coeff @ np.diag(mf.mo_energy) @ mf.mo_coeff.T @ s1e
    u, _ = correct_fock(model, mf.mo_energy, mf.mo_coeff, mf.mo_occ, np.zeros(gly.nao), d, max_rot=10.0)
    e2, c2 = mf.eig(fock + u, s1e)
    dD_fock = mf.make_rdm1(c2, mf.get_occ(e2, c2)) - D_R
    assert np.linalg.norm(dD_fock - dD_rot) / np.linalg.norm(dD_rot) < 1e-3
