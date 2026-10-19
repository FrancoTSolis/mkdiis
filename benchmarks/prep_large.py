"""Geometries for the large closed-shell set and the ADIIS-paper substitutes -> data/large.json.

    python prep_large.py      (needs RDKit; reads the 1L2Y PDB file in data/)

Entries: name -> dict(atoms=[[El, x, y, z], ...] in Angstrom, charge, spin, point_charges=[[q, x, y, z], ...]).
  graphene C24H12, C54H18, C96H24   hexagonal flakes cut from a honeycomb lattice (C-C 1.42, C-H 1.09)
  thio6, thio10                     alpha-oligothiophenes, MMFF94 geometry
  h2tpp, zntpp                      tetraphenylporphyrin (MMFF94); Zn placed at the N4 centroid, NH removed
  ala<n>                            Ace-(Ala)n-NMe alpha helix (phi -57, psi -47), MMFF94-relaxed
  trpcage                           1L2Y model 1 (with H) + unit counter-charges 3.5 A outside each charged group
  ch3cho, water51, ru1, ru2         substitutes for the ADIIS paper's test molecules (its geometries are
                                    no longer online): two closed-shell Ru complexes with 30-60 atoms from tmQM
"""
import json
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

HERE = os.path.dirname(os.path.abspath(__file__))


def flake(nring):
    """Hexagonal flake: hexagons whose centres lie within nring - 1 shells of the central one."""
    a = 1.42
    a1, a2 = np.array([1.5 * a, np.sqrt(3) / 2 * a]), np.array([0.0, np.sqrt(3) * a])
    centres = []
    for i in range(-nring, nring + 1):
        for j in range(-nring, nring + 1):
            k = -i - j
            if max(abs(i), abs(j), abs(k)) <= nring - 1:
                centres.append(i * a1 + j * a2)
    verts = {}
    for c in centres:
        for t in range(6):
            p = c + a * np.array([np.cos(np.pi / 3 * t), np.sin(np.pi / 3 * t)])
            verts[tuple(np.round(p, 4))] = p
    C = np.array(list(verts.values()))
    d = np.linalg.norm(C[:, None] - C[None], axis=-1)
    atoms = [["C", x, y, 0.0] for x, y in C]
    for i in range(len(C)):
        nb = np.where((d[i] > 0.1) & (d[i] < 1.6))[0]
        if len(nb) == 2:
            u = 2 * C[i] - C[nb].sum(0)
            u /= np.linalg.norm(u)
            atoms.append(["H", *(C[i] + 1.09 * u), 0.0])
    return atoms


def to_atoms(m, conf_id=-1):
    pos = m.GetConformer(conf_id).GetPositions()
    return [[a.GetSymbol(), *map(float, p)] for a, p in zip(m.GetAtoms(), pos)]


def embed(smiles, seed=7, opt=True):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    if opt:
        AllChem.MMFFOptimizeMolecule(m, maxIters=5000)
    return m


def helix(n):
    seq = "CC(=O)" + "N[C@@H](C)C(=O)" * n + "NC"
    m = Chem.AddHs(Chem.MolFromSmiles(seq))
    params = AllChem.ETKDGv3()
    params.randomSeed, params.useRandomCoords = 11, True
    if AllChem.EmbedMolecule(m, params) < 0:
        raise RuntimeError(f"embedding failed for ala{n}")
    conf = m.GetConformer()
    patt = Chem.MolFromSmarts("C(=O)N[C@@H](C)C(=O)N")
    for match in m.GetSubstructMatches(patt):
        c0, _, n, ca, _, c, _, n1 = match
        rdMolTransforms.SetDihedralDeg(conf, c0, n, ca, c, -57.0)
        rdMolTransforms.SetDihedralDeg(conf, n, ca, c, n1, -47.0)
    for match in m.GetSubstructMatches(Chem.MolFromSmarts("[CX4][CX3](=O)[NX3][CX4]")):
        rdMolTransforms.SetDihedralDeg(conf, *match[:2], match[3], match[4], 180.0)
    mp = AllChem.MMFFGetMoleculeProperties(m)
    ff = AllChem.MMFFGetMoleculeForceField(m, mp)
    for match in m.GetSubstructMatches(patt):
        c0, _, n, ca, _, c, _, n1 = match
        ff.MMFFAddTorsionConstraint(c0, n, ca, c, False, -57.0, -57.0, 100.0)
        ff.MMFFAddTorsionConstraint(n, ca, c, n1, False, -47.0, -47.0, 100.0)
    ff.Minimize(maxIts=5000)
    return m


def tpp(zinc):
    m = embed("c1ccc(cc1)-c1c2ccc(n2)c(-c2ccccc2)c2ccc([nH]2)c(-c2ccccc2)c2ccc(n2)c(-c2ccccc2)c2ccc1[nH]2")
    atoms = to_atoms(m)
    if not zinc:
        return atoms, 0
    nh = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "N"]
    hs = [n.GetIdx() for i in nh for n in m.GetAtomWithIdx(i).GetNeighbors() if n.GetSymbol() == "H"]
    zn = np.mean([atoms[i][1:] for i in nh], axis=0)
    return [a for i, a in enumerate(atoms) if i not in hs] + [["Zn", *map(float, zn)]], 0


def trpcage(pdb):
    """Model 1 of 1L2Y with unit counter-charges placed 3.5 A outside each charged group."""
    lines = [ln for ln in open(pdb) if ln.startswith(("ATOM", "HETATM", "ENDMDL"))]
    atoms, names = [], []
    for ln in lines:
        if ln.startswith("ENDMDL"):
            break
        el = ln[76:78].strip() or ln[12:16].strip()[0]
        atoms.append([el.capitalize(), float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        names.append((ln[17:20], int(ln[22:26]), ln[12:16].strip()))
    xyz = np.array([a[1:] for a in atoms])
    centre = xyz.mean(0)
    sites = []
    resids = [n[1] for n in names]
    first, last = min(resids), max(resids)
    for i, (res, rid, nm) in enumerate(names):
        q = None
        if rid == first and nm == "N":
            q = -1.0
        elif rid == last and nm == "OXT":
            q = +1.0
        elif res == "LYS" and nm == "NZ":
            q = -1.0
        elif res == "ARG" and nm == "CZ":
            q = -1.0
        elif res == "ASP" and nm == "CG":
            q = +1.0
        elif res == "GLU" and nm == "CD":
            q = +1.0
        if q is not None:
            u = xyz[i] - centre
            u /= np.linalg.norm(u)
            sites.append([q, *map(float, xyz[i] + 3.5 * u)])
    nelec = sum(Chem.GetPeriodicTable().GetAtomicNumber(a[0]) for a in atoms)
    charge = int(round(sum(-s[0] for s in sites)))
    return atoms, charge, sites, nelec


def water_cluster(n, seed=0):
    """n waters on a 3.0 A cubic grid, random orientations without intermolecular H...H < 2.0 or O...H < 1.7 A."""
    rng = np.random.default_rng(seed)
    g = np.array([[i, j, k] for i in range(-4, 5) for j in range(-4, 5) for k in range(-4, 5)], float) * 3.0
    g = g[np.argsort(np.linalg.norm(g, axis=1), kind="stable")][:n]
    oh, ang = 0.9572, np.deg2rad(104.52)
    local = np.array([[0, 0, 0], [oh, 0, 0], [oh * np.cos(ang), oh * np.sin(ang), 0]])
    placed = []
    for p in g:
        for _ in range(1000):
            q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            w = local @ q.T + p
            if all(np.linalg.norm(w[1:, None] - o[None, 1:], axis=-1).min() > 2.0
                   and np.linalg.norm(w[1:] - o[0], axis=-1).min() > 1.7 and np.linalg.norm(o[1:] - w[0], axis=-1).min() > 1.7
                   for o in placed):
                break
        placed.append(w)
    return [[el, *map(float, x)] for w in placed for el, x in zip("OHH", w)]


def main():
    out = {}
    for name, nr in (("C24H12", 2), ("C54H18", 3), ("C96H24", 4)):
        out[name] = dict(atoms=flake(nr), charge=0, spin=0)
    for n in (6, 10):
        out[f"thio{n}"] = dict(atoms=to_atoms(embed("c1ccsc1" if n == 1 else "c1ccc(s1)" + "-c1ccc(s1)" * (n - 2) + "-c1cccs1")),
                               charge=0, spin=0)
    for name, zinc in (("h2tpp", False), ("zntpp", True)):
        atoms, chg = tpp(zinc)
        out[name] = dict(atoms=atoms, charge=chg, spin=0)
    for n in (10, 15, 20):
        out[f"ala{n}"] = dict(atoms=to_atoms(helix(n)), charge=0, spin=0)
    pdb = os.path.join(HERE, "data", "1L2Y.pdb")
    if os.path.exists(pdb):
        atoms, charge, sites, nelec = trpcage(pdb)
        out["trpcage"] = dict(atoms=atoms, charge=charge, spin=0, point_charges=sites)
    out["ch3cho"] = dict(atoms=to_atoms(embed("CC=O")), charge=0, spin=0)
    out["water51"] = dict(atoms=water_cluster(51), charge=0, spin=0)
    tmqm = "/tmp/fts_tmqm"
    if os.path.isdir(tmqm):
        import glob
        from prep_tmqm import read
        ru = [e for p in sorted(glob.glob(os.path.join(tmqm, "tmQM_X*.xyz.gz"))) for e in read(p)
              if e["spin"] == 0 and 30 <= len(e["atoms"]) <= 60 and sum(a[0] == "Ru" for a in e["atoms"]) == 1]
        rng = np.random.default_rng(0)
        for k, i in enumerate(sorted(rng.choice(len(ru), size=2, replace=False))):
            e = ru[i]
            out[f"ru{k + 1}"] = dict(atoms=e["atoms"], charge=e["charge"], spin=0, csd=e["name"], formula=e["formula"])
    for k, v in out.items():
        els = [a[0] for a in v["atoms"]]
        print(f"{k:8s} natm {len(els):4d}  " + " ".join(f"{e}{els.count(e)}" for e in sorted(set(els)))
              + (f"  charge {v['charge']}" if v["charge"] else "") + (f"  {len(v.get('point_charges', []))} point charges" if v.get("point_charges") else ""))
    with open(os.path.join(HERE, "data", "large.json"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
