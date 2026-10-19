"""End-to-end charge sloshing in the Ace-(Ala)n-NMe alpha helix, from bench.py --trace runs (B3LYP/def2-SVP).

    python helix_figure.py [ala10 [dir]]   # reads <dir>/large_<name>__{diis,mk_spd_cg}.json, written by
                                           # bench.py --trace; dir defaults to data/

writes out/<name>_before_after.mp4 (key frames in out/frames_<name>_before_after/) and
out/<name>_static.{png,pdf}.  Every atom is colored by the error in the electron count of its residue (Mulliken
populations) against the converged density; the dipole along the helix axis measures the end-to-end charge transfer.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from anim_sloshing import BLUE, CMAP, GREY, INK, MUTED, SIZE, TEXT, X0, W, keyframes, render, schedule

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLIP = 0.05


def residues(sym, xyz, n):
    """Residue index of every atom: 0 = Ace, 1..n = Ala, n + 1 = NMe (heavy atoms in SMILES order, then H)."""
    heavy = [i for i, s in enumerate(sym) if s != "H"]
    res = np.empty(len(sym), int)
    for k, i in enumerate(heavy):
        res[i] = 0 if k < 3 else (n + 1 if k >= 3 + 5 * n else 1 + (k - 3) // 5)
    for i, s in enumerate(sym):
        if s == "H":
            res[i] = res[heavy[int(np.argmin(np.linalg.norm(xyz[heavy] - xyz[i], axis=1)))]]
    return res


def load(name, src=os.path.join(HERE, "data")):
    with open(os.path.join(ROOT, "benchmarks", "data", "large.json")) as f:
        atoms = json.load(f)[name]["atoms"]
    sym = [a[0] for a in atoms]
    xyz = np.array([a[1:] for a in atoms], float)
    n = int(name[3:])
    res = residues(sym, xyz, n)
    heavy = np.array([s != "H" for s in sym])
    c = xyz[heavy].mean(0)
    _, _, vt = np.linalg.svd(xyz[heavy] - c, full_matrices=False)
    axis = vt[0] if (xyz[res == n + 1].mean(0) - xyz[res == 0].mean(0)) @ vt[0] > 0 else -vt[0]
    plane = np.stack([axis, vt[1]])
    runs = {}
    for m in ("diis", "mk_spd_cg"):
        with open(os.path.join(src, f"large_{name}__{m}.json")) as f:
            runs[m] = json.load(f)
    ref_q = np.array(runs["mk_spd_cg"]["trace"]["charges"][-1])
    ref_mu = np.array(runs["mk_spd_cg"]["trace"]["dipole"][-1]) @ axis
    data = dict(name=name, n=n, sym=sym, xy=(xyz - c) @ plane.T, res=res, nres=n + 2)
    group = np.array([(res == k).astype(float) for k in range(n + 2)])
    for m, r in runs.items():
        q = np.array(r["trace"]["charges"])
        mu = np.array(r["trace"]["dipole"]) @ axis
        data[m] = dict(err=-(q - ref_q) @ group.T, mu=mu / ref_mu, grad=np.array([h[2] for h in r["hist"]]),
                       nfock=r["nfock"], e=r["e_tot"], wall=r["wall"])
    return data


class Helix:
    def __init__(self, fig, box, data, clip=CLIP):
        self.norm = Normalize(-clip, clip)
        self.res = data["res"]
        xy, sym = data["xy"], data["sym"]
        heavy = np.array([s != "H" for s in sym])
        ax = fig.add_axes(box)
        full = np.c_[xy, np.zeros(len(sym))]
        d = np.linalg.norm(full[:, None] - full[None], axis=-1)
        bonds = [(i, j) for i in range(len(sym)) for j in range(i)
                 if d[i, j] < (1.15 if "H" in (sym[i], sym[j]) else 1.7)]
        ax.add_collection(LineCollection([xy[[i, j]] for i, j in bonds], colors="#c3c8cf", lw=1.2, zorder=1))
        ax.scatter(xy[~heavy, 0], xy[~heavy, 1], s=14, fc="#e6e8eb", ec="#9aa1ab", lw=0.4, zorder=2)
        self.heavy = heavy
        self.dots = ax.scatter(xy[heavy, 0], xy[heavy, 1], s=70, ec="#5b6370", lw=0.6, zorder=3)
        pad = 1.2
        ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
        ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(xy[self.res == 0, 0].mean(), xy[:, 1].min() - 1.0, "N-terminus", ha="center", va="top",
                fontsize=11, color=MUTED)
        ax.text(xy[self.res == self.res.max(), 0].mean(), xy[:, 1].min() - 1.0, "C-terminus", ha="center",
                va="top", fontsize=11, color=MUTED)

    def update(self, err_res):
        self.dots.set_facecolor(CMAP(self.norm(err_res[self.res[self.heavy]])))


def make_animation(data):
    dd, dm = data["diis"], data["mk_spd_cg"]
    nf_d, nf_m = dd["nfock"], dm["nfock"]
    n_steps = nf_d - 1
    fig = plt.figure(figsize=SIZE)
    fig.text(0.05, 0.945, f"End-to-end sloshing in an \u03b1 helix: Ace-(Ala)$_{{{data['n']}}}$-NMe", fontsize=23,
             fontweight="bold", color=INK)
    fig.text(0.05, 0.905, "PySCF with our MK-DIIS code, B3LYP/def2-SVP, same starting guess. Every atom is colored by "
             "the electron error of its residue.", fontsize=13, color=MUTED)
    helices, status, notes = [], [], []
    for j, (name, sub, col) in enumerate((
            ("Standard DIIS", "charge sloshes between the two ends", GREY),
            ("MK-DIIS (ours)", "the model predicts the push-back", BLUE))):
        x0 = 0.05 + j * 0.48
        fig.text(x0, 0.845, name, fontsize=17, fontweight="bold", color=col, va="center")
        fig.text(x0, 0.81, sub, fontsize=12, color=MUTED, va="center")
        status.append(fig.text(x0 + 0.42, 0.845, "", ha="right", va="center", fontsize=14, fontweight="bold", color=col))
        helices.append(Helix(fig, [x0, 0.44, 0.42, 0.35], data))
        notes.append(fig.text(x0 + 0.21, 0.405, "", ha="center", va="center", fontsize=12.5, fontweight="bold",
                              color=col))
    cax = fig.add_axes([0.415, 0.2, 0.17, 0.022])
    cax.imshow(np.linspace(-CLIP, CLIP, 256)[None, :], aspect="auto", cmap=CMAP, norm=Normalize(-CLIP, CLIP),
               extent=(-CLIP, CLIP, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([-CLIP, 0, CLIP])
    cax.set_xticklabels([f"\u2264 \u2212{CLIP:g}", "0", f"\u2265 +{CLIP:g} e"], fontsize=10)
    for s in ("left", "top", "right"):
        cax.spines[s].set_visible(False)
    cax.set_title("electron error per residue\nblue too few, red too many", fontsize=10.5, color=TEXT, pad=4)

    ax = fig.add_axes([0.08, 0.11, 0.29, 0.23])
    ax.set_xlim(0.5, nf_d + 0.5)
    lo = min(dd["mu"].min(), dm["mu"].min(), 0.0)
    hi = max(dd["mu"].max(), dm["mu"].max(), 1.0)
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    ax.axhline(1.0, color=INK, lw=1.0, ls=(0, (3, 2)))
    ax.text(nf_d + 0.4, 1.0, "answer", ha="right", va="bottom", fontsize=10, color=INK)
    ax.set_xticks([1] + list(range(5, nf_d + 1, 5)))
    ax.set_xlabel("Fock builds")
    ax.set_ylabel("end-to-end charge\ntransfer (1 = answer)")
    mu_d, = ax.plot([], [], "o-", color=GREY, ms=3.5, lw=1.6)
    mu_m, = ax.plot([], [], "o-", color=BLUE, ms=3.5, lw=1.8)

    ax = fig.add_axes([0.67, 0.11, 0.29, 0.23])
    ax.set_yscale("log")
    ax.set_xlim(0, nf_d + 1.5)
    ax.set_ylim(1e-6, 30)
    ax.set_xlabel("Fock builds")
    ax.set_ylabel("orbital gradient norm")
    ax.axhline(np.sqrt(1e-9), color=MUTED, lw=0.9, ls=":")
    ax.text(0.3, np.sqrt(1e-9) * 1.4, "converged", ha="left", va="bottom", fontsize=10, color=MUTED)
    ln_d, = ax.plot([], [], "o-", color=GREY, ms=3.5, lw=1.6, label="DIIS")
    ln_m, = ax.plot([], [], "o-", color=BLUE, ms=3.5, lw=1.8, label="MK-DIIS")
    hd_d, = ax.plot([], [], "o", color=GREY, ms=8)
    hd_m, = ax.plot([], [], "o", color=BLUE, ms=8)
    ax.legend(loc="upper right", fontsize=11, handlelength=1.6)
    takeaway = fig.text(0.5, 0.37, f"same energy, {nf_m} vs {nf_d} Fock builds", ha="center", va="center",
                        fontsize=15, fontweight="bold", color=INK,
                        bbox=dict(boxstyle="round,pad=0.45", fc="#eef4fb", ec=BLUE, lw=1.2))

    def state(d, s):
        return d["err"][min(s, len(d["err"]) - 1)]

    def note(d, s):
        if s == 0 or s + 1 >= d["nfock"]:
            return ""
        mu = d["mu"][min(s, len(d["mu"]) - 1)]
        if abs(mu - 1) < 0.05:
            return "end-to-end charge transfer: the right amount"
        if mu < 0:
            return f"end-to-end charge transfer: {-mu:.1f}\u00d7 the right amount, wrong direction"
        if mu > 2:
            return f"end-to-end charge transfer: {mu:.1f}\u00d7 the right amount"
        return f"end-to-end charge transfer: {mu:.0%} of the right amount"

    def timing(k):
        return (16, 22) if k < 6 else (8, 8)

    frames = schedule(n_steps, timing, start_hold=40, end_hold=80)

    def draw(fr):
        a, b, t = fr
        shown = b if t > 0.5 else a
        for hx, d, st, nt, nf in ((helices[0], dd, status[0], notes[0], nf_d), (helices[1], dm, status[1], notes[1], nf_m)):
            hx.update((1 - t) * state(d, a) + t * state(d, b))
            st.set_text(f"converged \u2713  {nf} Fock builds" if shown + 1 >= nf else f"Fock builds: {shown + 1}")
            nt.set_text(note(d, shown))
        for ln, hd, mu, d, nf in ((ln_d, hd_d, mu_d, dd, nf_d), (ln_m, hd_m, mu_m, dm, nf_m)):
            m = min(shown, nf - 1)
            ln.set_data(np.arange(2, m + 2), d["grad"][:m])
            hd.set_data([m + 1], [d["grad"][m - 1]] if m > 0 else [np.nan])
            mu.set_data(np.arange(1, m + 2), d["mu"][: m + 1])
        takeaway.set_visible(shown + 1 >= nf_d)
        for nt in notes:
            nt.set_visible(shown + 1 < nf_d)

    names = {s: f"state{s:02d}" for s in (0, 1, 2, 3, 4, 5, nf_m - 1, n_steps)}
    render(fig, frames, draw, f"{data['name']}_before_after", keyframes(frames, names))
    plt.close(fig)


def make_static(data):
    fig = plt.figure(figsize=(12.8, 5.6))
    rows = data["diis"]["nfock"]
    labels = ["Ace"] + [str(k) for k in range(1, data["n"] + 1)] + ["NMe"]
    for j, (m, title, col) in enumerate((("diis", "Standard DIIS", GREY), ("mk_spd_cg", "MK-DIIS (ours)", BLUE))):
        d = data[m]
        ax = fig.add_axes([0.07 + j * 0.25, 0.21, 0.21, 0.6])
        img = np.full((rows, data["nres"]), np.nan)
        img[: len(d["err"])] = d["err"]
        ax.imshow(img, aspect="auto", cmap=CMAP, vmin=-CLIP, vmax=CLIP, interpolation="nearest",
                  extent=(-0.5, data["nres"] - 0.5, rows + 0.5, 0.5))
        if d["nfock"] < rows:
            ax.axhspan(d["nfock"] + 0.5, rows + 0.5, color="#f3f4f6", zorder=2)
            ax.text((data["nres"] - 1) / 2, (d["nfock"] + rows) / 2 + 0.5, "converged", ha="center", va="center",
                    fontsize=12, color=BLUE, fontweight="bold", zorder=3)
        ax.text(0, 1.085, title, transform=ax.transAxes, fontsize=13, fontweight="bold", color=col)
        ax.text(0, 1.015, f"converged after {d['nfock']} Fock builds", transform=ax.transAxes, fontsize=11, color=col)
        ticks = [0, data["nres"] // 2, data["nres"] - 1]
        ax.set_yticks([1] + list(range(5, rows + 1, 5)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[t] for t in ticks])
        ax.set_xlabel("residue (N- to C-terminus)")
        if j == 0:
            ax.set_ylabel("Fock build")
        ax.text(-0.14, 1.03, "ab"[j], transform=ax.transAxes, fontsize=14, fontweight="bold", color=INK)
    cax = fig.add_axes([0.07, 0.04, 0.21, 0.025])
    cax.imshow(np.linspace(-CLIP, CLIP, 256)[None, :], aspect="auto", cmap=CMAP, vmin=-CLIP, vmax=CLIP,
               extent=(-CLIP, CLIP, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([-CLIP, 0, CLIP])
    cax.set_xticklabels([f"\u2264 \u2212{CLIP:g}", "0", f"\u2265 {CLIP:g}"], fontsize=9.5)
    for s in ("left", "top", "right"):
        cax.spines[s].set_visible(False)
    fig.text(0.3, 0.052, "electron error per residue (Mulliken): blue too few, red too many", fontsize=10.5,
             color=TEXT, va="center")

    ax = fig.add_axes([0.62, 0.21, 0.17, 0.6])
    for m, col, lab in (("diis", GREY, "DIIS"), ("mk_spd_cg", BLUE, "MK-DIIS")):
        g = data[m]["grad"]
        ax.semilogy(np.arange(2, len(g) + 2), g, "o-", color=col, ms=3.5, lw=1.6, label=lab)
    ax.axhline(np.sqrt(1e-9), color=MUTED, lw=0.9, ls=":")
    ax.set_xlabel("Fock builds")
    ax.set_ylabel("orbital gradient norm")
    ax.legend(loc="upper right", fontsize=10.5)
    ax.text(-0.3, 1.03, "c", transform=ax.transAxes, fontsize=14, fontweight="bold", color=INK)

    ax = fig.add_axes([0.86, 0.21, 0.12, 0.6])
    for m, col in (("diis", GREY), ("mk_spd_cg", BLUE)):
        mu = data[m]["mu"]
        ax.plot(np.arange(1, len(mu) + 1), mu, "o-", color=col, ms=3.5, lw=1.6)
    ax.axhline(1.0, color=INK, lw=0.9, ls=(0, (3, 2)))
    ax.axhline(0.0, color=MUTED, lw=0.6)
    ax.set_xlim(0.5, min(rows, 12) + 0.5)
    ax.set_xlabel("Fock builds")
    ax.set_ylabel("dipole along the helix axis\n(1 = converged)")
    ax.text(-0.45, 1.03, "d", transform=ax.transAxes, fontsize=14, fontweight="bold", color=INK)

    fig.text(0.07, 0.955, f"Ace-(Ala)$_{{{data['n']}}}$-NMe, B3LYP/def2-SVP: charge sloshes between the ends of the helix "
             "under DIIS, not under MK-DIIS", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.07, 0.918, "Real PySCF runs with our code from the same MINAO guess; both reach the same energy "
             f"({data['diis']['e']:.6f} vs {data['mk_spd_cg']['e']:.6f} Eh).", fontsize=10.5, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "out", f"{data['name']}_static.{ext}"), dpi=150)
    print(f"wrote out/{data['name']}_static.{{png,pdf}}")


if __name__ == "__main__":
    args = sys.argv[1:]
    data = load(args[0] if args else "ala10", *args[1:2])
    for m in ("diis", "mk_spd_cg"):
        print(m, data[m]["nfock"], "builds; axial dipole / converged:", " ".join(f"{x:+.2f}" for x in data[m]["mu"][:10]))
    make_static(data)
    make_animation(data)
