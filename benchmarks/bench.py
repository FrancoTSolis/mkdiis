"""Run one SCF strategy on one system and write a JSON record (the driver behind the paper's benchmarks).

    python bench.py --system tmqm:5:singlet --method mk_spd --out results/tmqm/tmqm_5_singlet__mk_spd.json

Systems:   tmqm:<index>:<state>     tmQM complex from data/tmqm_small.json, B3LYP/def2-SVP, state singlet|triplet
           organic:<index>          PubChem molecule from data/organic.json, B3LYP/def2-SVP
           radical:<index>          the same molecule without its last hydrogen atom, a doublet (UKS)
           acene:<n>                linear acene with n rings, PBE/def2-SVP
           ico:<El>:<charge>        13-atom icosahedral cluster, PBE/def2-SVP
           large:<name>[:<xc>[:<basis>]]  entry of data/large.json, default B3LYP/def2-SVP; point charges in the
                                    entry (Trp-cage counter-charges) are added as an external potential
           cdimid:<guess>           Cd(II)-imidazole, B3LYP/3-21G (the Q-Chem manual's ADIIS example)
           --xc and --basis override the functional ("hf" for Hartree-Fock) and basis of any system.
Methods:   diis        this driver without correction (reproduces PySCF's DIIS cycle by cycle)
           mk_spd      MK(spd), the default model; mk_s: MK(s); mk_jfit: MK(J-fit), def2-universal-jfit
           mk_spd_cg   MK(spd) with the AO-direct conjugate-gradient solver, for the large molecules
           mk_spd_k    MK(spd) plus the hybrid's exact exchange in the same auxiliary basis (MINRES solve)
           mk_exact    the exact-kernel bound (each product is a response build)
           adiis, adiis_<mk method>   ADIIS until the DIIS error drops below 1e-3, then DIIS / MK-DIIS
           pyscf, pyscf_adiis, pyscf_ediis   PySCF's own DIIS variants
           pyscf_soscf (second-order CIAH), pyscf_soscf_dfh (CIAH with the Hessian density-fitted in
           def2-universal-jfit)
           aurora, aurora_auxfull   AURORA-SCF of Bao and Shi (needs pyscf-aurora): adaptive curvature, and the
           valence-STO-3G Coulomb + exchange curvature of their paper
Cost is counted as njk, the number of J/K builds (Fock builds plus response builds).
"""
import argparse
import json
import os
import time

import numpy as np
from pyscf import dft, gto, lib, scf

import mkdiis

HERE = os.path.dirname(os.path.abspath(__file__))

CDIMID = """Cd 0.000000 0.000000 0.000000; N 0.000000 0.000000 -2.260001; N -0.685444 0.000000 -4.348035
C 0.676053 0.000000 -4.385069; C 1.085240 0.000000 -3.091231; C -1.044752 0.000000 -3.060220
H 1.231530 0.000000 -5.300759; H 2.088641 0.000000 -2.711077; H -2.068750 0.000000 -2.726515
H -1.313170 0.000000 -5.174718"""

NN = {"Li": 3.04, "Na": 3.72, "Al": 2.86, "Cu": 2.56, "Ag": 2.89, "Au": 2.88, "Pt": 2.77, "Pd": 2.75, "Ni": 2.49}


def acene(n):
    """Linear acene with n rings (C-C 1.40, C-H 1.09 Angstrom)."""
    a, ch = 1.40, 1.09
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


def icosahedron(el):
    """13-atom icosahedron with the bulk nearest-neighbour distance of the element."""
    phi = (1 + 5 ** 0.5) / 2
    v = np.array([[0, s1, s2 * phi] for s1 in (-1, 1) for s2 in (-1, 1)], float)
    v = np.vstack([v, np.roll(v, 1, axis=1), np.roll(v, 2, axis=1)])
    v *= 0.951 * NN[el] / np.linalg.norm(v[0])
    return [(el, 0.0, 0.0, 0.0)] + [(el, *x) for x in v]


def make_system(spec):
    """Molecule, functional and initial guess for a system specification."""
    parts = spec.split(":")
    if parts[0] == "cdimid":
        mol = gto.M(atom=CDIMID, basis="3-21g", charge=2, spin=0, verbose=0)
        return mol, "b3lyp", parts[1]
    if parts[0] == "tmqm":
        with open(os.path.join(HERE, "data", "tmqm_small.json")) as f:
            entry = json.load(f)[int(parts[1])]
        spin = {"singlet": 0, "triplet": 2}[parts[2]]
        mol = gto.M(atom=entry["atoms"], basis="def2-svp", ecp="def2-svp", charge=entry["charge"], spin=spin, verbose=0)
        return mol, "b3lyp", "minao"
    if parts[0] == "organic":
        with open(os.path.join(HERE, "data", "organic.json")) as f:
            entry = json.load(f)[int(parts[1])]
        mol = gto.M(atom=entry["atoms"], basis="def2-svp", charge=entry["charge"], spin=0, verbose=0)
        return mol, "b3lyp", "minao"
    if parts[0] == "radical":
        with open(os.path.join(HERE, "data", "organic.json")) as f:
            entry = json.load(f)[int(parts[1])]
        atoms = entry["atoms"]
        last_h = max(i for i, a in enumerate(atoms) if a[0] == "H")
        mol = gto.M(atom=atoms[:last_h] + atoms[last_h + 1:], basis="def2-svp", charge=entry["charge"], spin=1,
                    verbose=0)
        return mol, "b3lyp", "minao"
    if parts[0] == "acene":
        mol = gto.M(atom=acene(int(parts[1])), basis="def2-svp", verbose=0)
        return mol, "pbe", "minao"
    if parts[0] == "ico":
        el, chg = parts[1], int(parts[2])
        nelec = 13 * gto.charge(el) - chg
        mol = gto.M(atom=icosahedron(el), basis="def2-svp", ecp="def2-svp", charge=chg, spin=nelec % 2, verbose=0)
        return mol, "pbe", "minao"
    if parts[0] == "large":
        with open(os.path.join(HERE, "data", "large.json")) as f:
            entry = json.load(f)[parts[1]]
        heavy = any(gto.charge(a[0]) > 36 for a in entry["atoms"])
        basis = parts[3] if len(parts) > 3 else "def2-svp"
        mol = gto.M(atom=entry["atoms"], basis=basis, ecp=basis if heavy else None, charge=entry["charge"],
                    spin=entry["spin"], verbose=0)
        mol.point_charges = entry.get("point_charges")
        return mol, parts[2] if len(parts) > 2 else "b3lyp", "minao"
    raise ValueError(spec)


def override_basis(mol, basis):
    """Same molecule in another basis; the def2 pseudopotentials are the same for every def2 basis."""
    ecp = mol.ecp
    if ecp:
        ecp = "def2-svp" if basis.lower().startswith("def2") else basis
    out = gto.M(atom=mol._atom, unit="Bohr", basis=basis, ecp=ecp, charge=mol.charge, spin=mol.spin, verbose=0)
    out.point_charges = getattr(mol, "point_charges", None)
    return out


def make_mf(mol, xc, max_memory=3000, solvent=None):
    """RKS/UKS (RHF/UHF for xc="hf"); PBE runs density-fitted in def2-universal-jfit, as in the paper."""
    if xc.lower() == "hf":
        mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
    else:
        mf = dft.RKS(mol, xc=xc) if mol.spin == 0 else dft.UKS(mol, xc=xc)
    if xc == "pbe":
        mf = mf.density_fit()
    if solvent:
        mf = mf.PCM()
        mf.with_solvent.method = "IEF-PCM"
        mf.with_solvent.eps = {"water": 78.3553}[solvent]
    pc = getattr(mol, "point_charges", None)
    if pc:
        from pyscf import qmmm
        pc = np.asarray(pc)
        mf = qmmm.mm_charge(mf, pc[:, 1:], pc[:, 0], unit="Angstrom")
    mf.max_memory = max_memory
    return mf


def make_model(key, args):
    """(model, solver) of an MK method name."""
    if key == "diis":
        return None, "lowrank"
    if key == "mk_exact":
        return "exact", "lowrank"
    models = {"mk_s": lambda: mkdiis.AuxHartreeModel("s", theta=args.theta),
              "mk_spd": lambda: mkdiis.AuxHartreeModel("spd", theta=args.theta),
              "mk_jfit": lambda: mkdiis.AuxHartreeModel("def2-universal-jfit"),
              "mk_spd_k": lambda: mkdiis.AuxHartreeModel("spd", theta=args.theta, exchange=True),
              "mk_spd_cg": lambda: mkdiis.AuxHartreeModel("spd", theta=args.theta, store="tril", screen=args.screen)}
    return models[key](), "cg" if key.endswith("_cg") else "lowrank"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_cycle", type=int, default=150)
    ap.add_argument("--max_rot", type=float, default=0.3)
    ap.add_argument("--theta", type=float, default=0.2)
    ap.add_argument("--off_grad", type=float, default=0.0, help="apply the model only while |g| > off_grad")
    ap.add_argument("--on_grad", type=float, default=float("inf"), help="apply the model only once |g| < on_grad")
    ap.add_argument("--stab", type=int, default=0, help="rounds of internal-stability check + restart (0 = none)")
    ap.add_argument("--stab_mode", default="block", help="block (multi-start Davidson) or pyscf (PySCF's check)")
    ap.add_argument("--guess", default=None, help="override the initial guess (minao, atom, sap, huckel, core)")
    ap.add_argument("--smearing", type=float, default=0.0, help="Fermi-Dirac width in Hartree (this driver only)")
    ap.add_argument("--model_form", default="rotate", help="rotate (orbital rotation) or fock (Fock correction)")
    ap.add_argument("--no_intraband", action="store_true", help="with smearing, drop the f' and dmu terms of chi0")
    ap.add_argument("--cg_rtol", type=float, default=1e-3)
    ap.add_argument("--screen", type=float, default=1e-10, help="pair screening of the CG model (0 = none)")
    ap.add_argument("--max_memory", type=int, default=3000, help="PySCF max_memory in MB")
    ap.add_argument("--lindep", type=float, default=0.0,
                    help="drop AO combinations with overlap eigenvalue below this (canonical orthogonalization)")
    ap.add_argument("--xc", default=None, help="override the functional of the system (hf = Hartree-Fock)")
    ap.add_argument("--basis", default=None, help="override the basis set of the system")
    ap.add_argument("--solvent", default=None, help="IEF-PCM solvent (water)")
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="restart test: converge the given geometry with PySCF, displace every atom by a Gaussian "
                         "of this width (Angstrom) and start from the converged orbitals")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", action="store_true",
                    help="store the Mulliken charges and the dipole moment of every Fock-build density")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    lib.num_threads(args.threads)

    mol, xc, guess = make_system(args.system)
    xc = args.xc or xc
    if args.basis:
        mol = override_basis(mol, args.basis)
    guess = args.guess or guess
    restart = None
    if args.perturb > 0:
        mf0 = make_mf(mol, xc, args.max_memory, args.solvent)
        mf0.conv_tol = 1e-10
        mf0.kernel()
        rng = np.random.default_rng(args.seed)
        shift = rng.normal(scale=args.perturb, size=(mol.natm, 3))
        mol = mol.set_geom_(mol.atom_coords(unit="Angstrom") + shift, unit="Angstrom", inplace=False)
        restart = dict(perturb=args.perturb, seed=args.seed, ref_converged=bool(mf0.converged),
                       rms_shift=float(np.sqrt((shift ** 2).mean())))
        guess = "restart"
    mf = make_mf(mol, xc, args.max_memory, args.solvent)
    if args.lindep > 0:
        mf = scf.addons.remove_linear_dep_(mf, threshold=args.lindep, lindep=args.lindep)
    if restart is not None:
        s1 = mol.intor("int1e_ovlp")

        def occupied(c, occ):
            co = c[:, occ > 0]
            w, v = np.linalg.eigh(co.T @ s1 @ co)
            return co @ (v / np.sqrt(w)) @ v.T

        if mol.spin == 0:
            co = occupied(mf0.mo_coeff, mf0.mo_occ)
            dm0 = 2.0 * co @ co.T
        else:
            dm0 = np.array([(lambda co: co @ co.T)(occupied(mf0.mo_coeff[k], mf0.mo_occ[k])) for k in range(2)])
    else:
        dm0 = mf.get_init_guess(key={"core": "1e"}.get(guess, guess))
    njk = [0]
    for name in ("get_jk", "get_j", "get_k"):
        fn = getattr(mf, name)

        def counted(*a, _fn=fn, **k):
            njk[0] += 1
            return _fn(*a, **k)
        setattr(mf, name, counted)
    trace = None
    if args.trace:
        trace = dict(charges=[], dipole=[])
        s_ao = mol.intor("int1e_ovlp")
        r_ao = mol.intor("int1e_r", comp=3)
        ao_atom = np.zeros(mol.nao, int)
        for i, (_, _, a0, a1) in enumerate(mol.aoslice_by_atom()):
            ao_atom[a0:a1] = i
        veff = mf.get_veff

        def traced_veff(*a, **k):
            d = np.asarray(a[1] if len(a) > 1 else k["dm"])
            d = d if d.ndim == 2 else d[0] + d[1]
            pop = np.bincount(ao_atom, weights=np.einsum("ij,ji->i", d, s_ao), minlength=mol.natm)
            trace["charges"].append((mol.atom_charges() - pop).round(6).tolist())
            dip = mol.atom_charges() @ mol.atom_coords() - np.einsum("xij,ji->x", r_ao, d)
            trace["dipole"].append(dip.round(6).tolist())
            return veff(*a, **k)
        mf.get_veff = traced_veff
    t0 = time.perf_counter()
    rec = dict(system=args.system, method=args.method, natm=mol.natm, nao=mol.nao, nelec=mol.nelec,
               xc=xc, basis=str(mol.basis), guess=guess, max_cycle=args.max_cycle, solvent=args.solvent,
               restart=restart)
    if args.method.startswith("aurora"):
        import pyscf_aurora  # noqa: F401  (registers mf.aurora)
        mf = mf.aurora()
        if args.method == "aurora_auxfull":
            mf.aurora_hessian_fidelity = "aux-full"
        # pyscf-aurora 0.10.0 builds its valence-STO-3G basis before its fallback loop, so elements without STO-3G
        # (the 5d metals) raise instead of falling back to MINAO; select that fallback directly for them.
        try:
            for symb in {mol.atom_symbol(i) for i in range(mol.natm)}:
                gto.basis.load("sto-3g", symb)
        except Exception:
            mf.aurora_auxbasis = "minao"
        mf.max_cycle = args.max_cycle
        mf.conv_tol = 1e-9
        mf.kernel(dm0)
        st = mf.aurora_stats
        rec.update(converged=bool(mf.converged), e_tot=float(mf.e_tot), cycles=njk[0] - 1, nfock=njk[0], nresp=0,
                   wall=time.perf_counter() - t0, t_model=float(st.get("curvature_build_time", 0.0))
                   + float(st.get("curvature_refresh_time", 0.0)),
                   aurora=dict(version=getattr(pyscf_aurora, "__version__", None),
                               auxbasis=getattr(mf, "aurora_auxbasis", "auto"),
                               backend_jk_calls=st.get("backend_jk_calls"),
                               fidelity_final=st.get("hessian_fidelity_final"),
                               scf_wall_time=st.get("scf_wall_time")))
        dm, mo_c, mo_occ = mf.make_rdm1(), mf.mo_coeff, mf.mo_occ
    elif args.method.startswith("pyscf"):
        cyc = []
        if args.method == "pyscf_adiis":
            mf.DIIS = scf.diis.ADIIS
        elif args.method == "pyscf_ediis":
            mf.DIIS = scf.diis.EDIIS
        elif args.method == "pyscf_soscf":
            mf = mf.newton()
        elif args.method == "pyscf_soscf_dfh":
            mf = mf.newton().density_fit(auxbasis="def2-universal-jfit")
        mf.max_cycle = args.max_cycle
        mf.callback = lambda env: cyc.append(1)
        mf.kernel(dm0)
        rec.update(converged=bool(mf.converged), e_tot=float(mf.e_tot), cycles=len(cyc), nfock=len(cyc) + 1,
                   nresp=0, wall=time.perf_counter() - t0, t_model=0.0)
        dm, mo_c, mo_occ = mf.make_rdm1(), mf.mo_coeff, mf.mo_occ
    else:
        accel = "adiis" if args.method.startswith("adiis") else "cdiis"
        key = args.method[len("adiis_"):] if args.method.startswith("adiis_") else args.method
        key = "diis" if key == "adiis" else key
        model, solver = make_model(key, args)
        log = print if args.verbose else None
        kw = dict(max_cycle=args.max_cycle, max_rot=args.max_rot, accel=accel, model_off_grad=args.off_grad,
                  model_on_grad=args.on_grad, solver=solver, cg_rtol=args.cg_rtol, smearing=args.smearing,
                  model_form=args.model_form, intraband=not args.no_intraband, log=log)
        res = mkdiis.run_scf(mf, dm0, model=model, **kw)
        dm, mo_c, mo_occ = res.pop("dm"), res.pop("mo_coeff"), np.asarray(res.pop("mo_occ"))
        res.pop("fock")
        rounds = []
        for _ in range(args.stab):
            mo_new, stable = mkdiis.internal_stability(mf, mo_c, mo_occ, args.stab_mode)
            rounds.append(dict(e_tot=res["e_tot"], cycles=res["cycles"], stable=bool(stable)))
            if stable:
                break
            res2 = mkdiis.run_scf(mf, mf.make_rdm1(mo_new, mo_occ), model=model, **kw)
            dm, mo_c, mo_occ = res2.pop("dm"), res2.pop("mo_coeff"), np.asarray(res2.pop("mo_occ"))
            res2.pop("fock")
            res2["cycles"] += res["cycles"]
            res2["nfock"] += res["nfock"]
            for k in ("wall", "t_model", "t_fock"):
                res2[k] += res[k]
            res = res2
        res["stab_rounds"] = rounds
        rec.update(res)
        rec.update(max_rot=args.max_rot, theta=args.theta, off_grad=args.off_grad, on_grad=args.on_grad,
                   smearing=args.smearing, model_form=args.model_form, stab_mode=args.stab_mode,
                   intraband=not args.no_intraband)
        if solver == "cg":
            rec.update(cg_rtol=args.cg_rtol, screen=args.screen, npair_kept=int(model.npair), cond_j=model.cond_j)
    rec["njk"] = njk[0]
    if trace is not None:
        rec["trace"] = trace
    rec["threads"] = args.threads
    rec["lindep"] = args.lindep
    s = mol.intor("int1e_ovlp")
    rec["dipole"] = np.asarray(mf.dip_moment(mol, dm, unit="Debye", verbose=0)).tolist()
    if np.ndim(dm) == 3:
        occ = (mo_c[0][:, mo_occ[0] > 0], mo_c[1][:, mo_occ[1] > 0])
        rec["s2"] = float(scf.uhf.spin_square(occ, s)[0])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".part", "w") as f:
        json.dump(rec, f)
    os.replace(args.out + ".part", args.out)
    print(json.dumps({k: rec[k] for k in ("system", "method", "converged", "e_tot", "cycles", "nfock", "nresp", "wall")}))


if __name__ == "__main__":
    main()
