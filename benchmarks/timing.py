"""Cost of the pieces of one SCF cycle on a large system: Fock build, diagonalization, model build and correction.

    python timing.py large:C96H24 [--threads 16] [--woodbury]
"""
import argparse
import json
import time

from pyscf import lib

import bench
import mkdiis
from mkdiis.correction import correct_cg, correct_lowrank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--woodbury", action="store_true", help="also time the dense low-rank (Woodbury) solve")
    ap.add_argument("--max_memory", type=int, default=20000)
    args = ap.parse_args()
    lib.num_threads(args.threads)
    mol, xc, guess = bench.make_system(args.system)
    mf = bench.make_mf(mol, xc, args.max_memory)
    out = dict(system=args.system, natm=mol.natm, nao=mol.nao, nocc=mol.nelectron // 2, xc=xc, threads=args.threads)
    t = time.perf_counter()
    dm0 = mf.get_init_guess(key=guess)
    out["t_guess"] = time.perf_counter() - t
    t = time.perf_counter()
    vhf = mf.get_veff(mol, dm0)
    out["t_fock"] = time.perf_counter() - t
    s1e, h1e = mf.get_ovlp(), mf.get_hcore()
    fock = mf.get_fock(h1e, s1e, vhf, dm0)
    t = time.perf_counter()
    mo_e, mo_c = mf.eig(fock, s1e)
    out["t_eig"] = time.perf_counter() - t
    mo_occ = mf.get_occ(mo_e, mo_c)
    dD = mf.make_rdm1(mo_c, mo_occ) - dm0
    print(json.dumps(out), flush=True)

    model = mkdiis.AuxHartreeModel("spd", store="tril", screen=1e-10)
    t = time.perf_counter()
    model.build(mol)
    out.update(t_model_build=time.perf_counter() - t, naux=model.naux, npair_kept=int(model.npair),
               pair_fraction=model.npair / (mol.nao * (mol.nao + 1) / 2), T_gb=model.T.nbytes / 1e9)
    t = time.perf_counter()
    model.apply(dD)
    out["t_apply"] = time.perf_counter() - t
    for rtol in (1e-2, 1e-3):
        t = time.perf_counter()
        _, kmax, na = correct_cg(model, mo_e, mo_c, mo_occ, dD, rtol=rtol)
        out[f"t_cg_{rtol:g}"], out[f"napply_{rtol:g}"] = time.perf_counter() - t, na
    out["kmax"] = kmax
    print(json.dumps(out), flush=True)
    if args.woodbury:
        dense = mkdiis.AuxHartreeModel("spd")
        t = time.perf_counter()
        dense.build(mol)
        out["t_dense_build"] = time.perf_counter() - t
        t = time.perf_counter()
        correct_lowrank(dense, mo_e, mo_c, mo_occ, dD)
        out["t_woodbury"] = time.perf_counter() - t
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
