"""Select small tmQM complexes (fewer than 20 atoms, as in Sethio et al. 2024) into data/tmqm_small.json.

    python prep_tmqm.py <dir with tmQM_X1.xyz.gz ...> [--n 120] [--seed 0]
"""
import argparse
import glob
import gzip
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def read(path):
    with gzip.open(path, "rt") as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i])
        head = lines[i + 1]
        meta = dict(re.findall(r"(\w+)\s*=\s*([^|]+)", head))
        atoms = [ln.split()[:4] for ln in lines[i + 2:i + 2 + n]]
        yield dict(name=meta["CSD_code"].strip(), charge=int(meta["q"]), spin=int(meta["S"]),
                   formula=meta.get("Stoichiometry", "").strip(),
                   atoms=[[a[0], float(a[1]), float(a[2]), float(a[3])] for a in atoms])
        i += 2 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    small = [e for p in sorted(glob.glob(os.path.join(args.src, "tmQM_X*.xyz.gz"))) for e in read(p) if len(e["atoms"]) < 20]
    print(f"{len(small)} complexes with fewer than 20 atoms")
    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(len(small), size=min(args.n, len(small)), replace=False).tolist())
    sel = [small[i] for i in pick]
    with open(os.path.join(HERE, "data", "tmqm_small.json"), "w") as f:
        json.dump(sel, f)
    metals = {}
    for e in sel:
        m = [a[0] for a in e["atoms"] if a[0] not in "H B C N O F Si P S Cl As Se Br I".split()]
        metals[m[0] if m else "?"] = metals.get(m[0] if m else "?", 0) + 1
    print(f"selected {len(sel)}; metals: {dict(sorted(metals.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    main()
