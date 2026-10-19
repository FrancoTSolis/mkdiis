import numpy as np
from conftest import sym
from pyscf import scf

from mkdiis import AuxHartreeModel, run_scf
from mkdiis.correction import _channels, hx_operator


def _rel(got, ref):
    return np.sqrt(sum(np.linalg.norm(a - b) ** 2 for a, b in zip(got, ref)) / sum(np.linalg.norm(b) ** 2 for b in ref))


def test_ri_hartree_exchange_operator(gly, gly_rks, gly_triplet, gly_uks, rng):
    hyb = 0.2
    for mol, mf in ((gly, gly_rks), (gly_triplet, gly_uks)):
        restricted = mol.spin == 0
        model = AuxHartreeModel("def2-universal-jkfit").build(mol)
        chans = _channels(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
        kvec, rhs, _, _ = hx_operator(model, chans, restricted, hyb)
        xs = [rng.standard_normal((len(c[5]), len(c[4]))) * 1e-2 for c in chans]
        dd = sym(rng, mol.nao)
        if restricted:
            Co, Cv = chans[0][2], chans[0][3]
            d = 2.0 * Cv @ xs[0] @ Co.T
            vj, vk = mf.get_jk(mol, d + d.T)
            exact = [Cv.T @ (vj - 0.5 * hyb * vk) @ Co]
            vj, vk = mf.get_jk(mol, dd)
            exact_rhs, got_rhs = [Cv.T @ (vj - 0.5 * hyb * vk) @ Co], rhs(dd)
        else:
            dms = np.array([c[3] @ x @ c[2].T for c, x in zip(chans, xs)])
            vj, vk = mf.get_jk(mol, dms + dms.transpose(0, 2, 1))
            exact = [c[3].T @ (vj[0] + vj[1] - hyb * vk[s]) @ c[2] for s, c in enumerate(chans)]
            dd2 = np.array([dd, 0.4 * dd])
            vj, vk = mf.get_jk(mol, dd2)
            exact_rhs = [c[3].T @ (vj[0] + vj[1] - hyb * vk[s]) @ c[2] for s, c in enumerate(chans)]
            got_rhs = rhs(dd2)
        assert _rel(kvec(xs), exact) < 1e-2
        assert _rel(got_rhs, exact_rhs) < 1e-2


def test_hartree_fock_with_the_exchange_model(gly):
    ref = scf.RHF(gly)
    ref.kernel()
    mf = scf.RHF(gly)
    res = run_scf(mf, mf.get_init_guess(), model=AuxHartreeModel("spd", exchange=True))
    assert res["converged"] and abs(res["e_tot"] - ref.e_tot) < 1e-7
