"""Summarize a results directory: convergence, Fock builds and wall time relative to plain DIIS.

    python summarize.py results/tmqm [--base diis] [--group triplet]

A run counts as a success only if it converged to the lowest energy found for that system by any
method (within 1e-5 Ha); converging to a higher SCF solution is reported separately.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def cost(r):
    """J/K builds: Fock builds plus response builds."""
    return r["njk"] if "njk" in r else r["nfock"] + r.get("nresp", 0)


def load(d):
    runs = defaultdict(dict)
    for p in glob.glob(os.path.join(d, "*.json")):
        try:
            r = json.load(open(p))
        except json.JSONDecodeError:
            print(f"unreadable, skipped: {p}")
            continue
        runs[r["system"]][r["method"]] = r
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--base", default="diis")
    ap.add_argument("--group", default=None, help="substring filter on the system name, e.g. triplet")
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()
    runs = load(args.dir)
    if args.group:
        runs = {k: v for k, v in runs.items() if args.group in k}
    methods = sorted({m for v in runs.values() for m in v}, key=lambda m: (m != args.base, m))
    print(f"{len(runs)} systems; methods: {methods}")

    best = {}
    for s, v in runs.items():
        es = [r["e_tot"] for r in v.values() if r["converged"]]
        best[s] = min(es) if es else None

    def ok(s, m):
        r = runs[s].get(m)
        return r is not None and r["converged"] and best[s] is not None and r["e_tot"] - best[s] < args.tol

    print(f"\n{'method':10s} {'n':>4s} {'success':>8s} {'higher-sol':>10s} {'failed':>7s}")
    for m in methods:
        have = [s for s in runs if m in runs[s]]
        succ = sum(ok(s, m) for s in have)
        conv = sum(runs[s][m]["converged"] for s in have)
        print(f"{m:10s} {len(have):4d} {succ:8d} {conv - succ:10d} {len(have) - conv:7d}")

    print(f"\npaired vs {args.base} (systems where both reach the lowest solution):")
    print(f"{'method':10s} {'n':>4s} {'median builds':>14s} {'base':>6s} {'geo-mean ratio':>15s} {'fewer':>6s} {'more':>5s} "
          f"{'wall ratio':>10s} {'model share':>11s}")
    for m in methods:
        if m == args.base:
            continue
        pairs = [s for s in runs if m in runs[s] and args.base in runs[s] and ok(s, m) and ok(s, args.base)]
        if not pairs:
            continue
        nb = np.array([cost(runs[s][m]) for s in pairs], float)
        n0 = np.array([cost(runs[s][args.base]) for s in pairs], float)
        w = np.array([runs[s][m]["wall"] for s in pairs]) / np.array([runs[s][args.base]["wall"] for s in pairs])
        share = np.array([runs[s][m].get("t_model", 0.0) / runs[s][m]["wall"] for s in pairs])
        print(f"{m:10s} {len(pairs):4d} {np.median(nb):14.1f} {np.median(n0):6.1f} {np.exp(np.mean(np.log(nb / n0))):15.3f} "
              f"{int((nb < n0).sum()):6d} {int((nb > n0).sum()):5d} {np.exp(np.mean(np.log(w))):10.3f} {np.median(share):11.3f}")

    print(f"\nrescues / losses vs {args.base}:")
    for m in methods:
        if m == args.base:
            continue
        common = [s for s in runs if m in runs[s] and args.base in runs[s]]
        resc = [s for s in common if ok(s, m) and not ok(s, args.base)]
        lost = [s for s in common if ok(s, args.base) and not ok(s, m)]
        print(f"{m:10s} rescued {len(resc):3d}  lost {len(lost):3d}   lost: {lost[:6]}")


if __name__ == "__main__":
    main()
