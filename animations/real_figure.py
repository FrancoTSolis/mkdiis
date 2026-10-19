"""Real-molecule counterpart of the toy animations, from real_trace.py output (octacene, PBE/def2-SVP).

    python real_figure.py [data/trace_acene8.npz]

writes out/acene_before_after.mp4 (key frames in out/frames_acene_before_after/) and out/acene_static.{png,pdf}.
Each carbon is colored by the error in the electron count of its group (the carbon and its hydrogen, if any;
Mulliken populations) against the converged density, at each Fock build.  The slow mode of DIIS here is the
charge transfer between the CH groups on the edge and the inner (ring-fusion) carbons.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from anim_sloshing import BLUE, CMAP, DPI, GREY, INK, MUTED, SIZE, TEXT, X0, W, ease, keyframes, render, schedule

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP = 0.1
RADIUS = {"C": 0.46, "H": 0.3}


class Molecule:
    """Carbons colored by the error of their group; hydrogens drawn grey."""

    def __init__(self, fig, box, xyz, sym, clip=CLIP):
        self.norm = Normalize(-clip, clip)
        self.carbons = np.array([s == "C" for s in sym])
        ax = fig.add_axes(box)
        d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
        bonds = [(i, j) for i in range(len(sym)) for j in range(i) if d[i, j] < (1.2 if "H" in (sym[i], sym[j]) else 1.6)]
        ax.add_collection(LineCollection([xyz[[i, j], :2] for i, j in bonds], colors="#9aa1ab", lw=2.0, zorder=1))
        ax.scatter(xyz[~self.carbons, 0], xyz[~self.carbons, 1], s=(RADIUS["H"] * 34) ** 2, fc="#e6e8eb",
                   ec="#8c959f", lw=0.6, zorder=2)
        self.dots = ax.scatter(xyz[self.carbons, 0], xyz[self.carbons, 1], s=(RADIUS["C"] * 34) ** 2, ec="#5b6370",
                               lw=0.7, zorder=3)
        pad = 0.8
        ax.set_xlim(xyz[:, 0].min() - pad, xyz[:, 0].max() + pad)
        ax.set_ylim(xyz[:, 1].min() - pad, xyz[:, 1].max() + pad)
        ax.set_aspect("equal")
        ax.axis("off")

    def update(self, err_groups):
        self.dots.set_facecolor(CMAP(self.norm(err_groups)))


def load(path):
    """Errors in electrons (positive: too many) per carbon group, and the fraction t of the edge-to-inner
    charge transfer done at each build (0 at the guess, 1 at convergence)."""
    d = np.load(path)
    xyz, sym = d["coords"], [str(s) for s in d["symbols"]]
    carbons = [i for i, s in enumerate(sym) if s == "C"]
    owner = np.array([i if s == "C" else carbons[int(np.argmin(np.linalg.norm(xyz[carbons] - xyz[i], axis=1)))]
                      for i, s in enumerate(sym)])
    group = np.array([[(owner == c).astype(float) for c in carbons]])[0]          # (ncarbon, natom)
    edge = np.array([(owner == c).sum() > 1 for c in carbons])
    out = dict(xyz=xyz, sym=sym, carbons=carbons, edge=edge, x=xyz[carbons, 0])
    for m in ("diis", "mk"):
        err = -(d[f"{m}_charges"] - d["ref"]) @ group.T
        t = 1.0 - (err @ err[0]) / (err[0] @ err[0])
        out[m] = dict(err=err, grad=d[f"{m}_grad"], nfock=int(d[f"{m}_nfock"]), e=float(d[f"{m}_e"]), t=t)
    return out


def make_animation(data):
    dd, dm = data["diis"], data["mk"]
    nf_d, nf_m = dd["nfock"], dm["nfock"]
    n_steps = nf_d - 1
    fig = plt.figure(figsize=SIZE)
    fig.text(0.05, 0.945, "The same effect in a real molecule: octacene, C$_{34}$H$_{20}$", fontsize=23,
             fontweight="bold", color=INK)
    fig.text(0.05, 0.905, "PySCF with our MK-DIIS code, PBE/def2-SVP, same starting guess. Each carbon is colored by "
             "how far the charge of its CH group is from the answer.", fontsize=13, color=MUTED)
    mols, status, notes = [], [], []
    for j, (name, sub, col) in enumerate((
            ("Standard DIIS", "charge sloshes between the CH edge and the inner carbons", GREY),
            ("MK-DIIS (ours)", "the model predicts the push-back: no sloshing", BLUE))):
        top = 0.84 - j * 0.37
        fig.text(X0, top, name, fontsize=17, fontweight="bold", color=col, va="center")
        fig.text(X0, top - 0.034, sub, fontsize=12, color=MUTED, va="center")
        status.append(fig.text(X0 + W, top, "", ha="right", va="center", fontsize=14, fontweight="bold", color=col))
        mols.append(Molecule(fig, [X0, top - 0.29, W, 0.23], data["xyz"], data["sym"]))
        notes.append(fig.text(X0 + W / 2, top - 0.305, "", ha="center", va="center", fontsize=12.5,
                              fontweight="bold", color=col))

    cax = fig.add_axes([X0, 0.06, 0.2, 0.022])
    cax.imshow(np.linspace(-CLIP, CLIP, 256)[None, :], aspect="auto", cmap=CMAP, norm=Normalize(-CLIP, CLIP),
               extent=(-CLIP, CLIP, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([-CLIP, 0, CLIP])
    cax.set_xticklabels([f"\u2264 \u2212{CLIP:g}", "0", f"\u2265 +{CLIP:g} e"], fontsize=10)
    for s in ("left", "top", "right"):
        cax.spines[s].set_visible(False)
    cax.set_title("electron error per group: blue too few, red too many", fontsize=10.5, color=TEXT, pad=4)

    ax = fig.add_axes([0.785, 0.2, 0.19, 0.6])
    ax.set_yscale("log")
    ax.set_xlim(0, nf_d + 1.5)
    ax.set_ylim(1e-6, 3)
    ax.set_xlabel("Fock builds")
    ax.set_ylabel("orbital gradient norm")
    ax.axhline(np.sqrt(1e-9), color=MUTED, lw=0.9, ls=":")
    ax.text(0.3, np.sqrt(1e-9) * 1.4, "converged", ha="left", va="bottom", fontsize=10, color=MUTED)
    ln_d, = ax.plot([], [], "o-", color=GREY, ms=3.5, lw=1.6, label="DIIS")
    ln_m, = ax.plot([], [], "o-", color=BLUE, ms=3.5, lw=1.8, label="MK-DIIS")
    hd_d, = ax.plot([], [], "o", color=GREY, ms=8)
    hd_m, = ax.plot([], [], "o", color=BLUE, ms=8)
    ax.legend(loc="upper right", fontsize=11.5, handlelength=1.6)
    takeaway = fig.text(0.88, 0.085, f"same energy,\n{nf_m} vs {nf_d} Fock builds", ha="center", va="center",
                        fontsize=15, fontweight="bold", color=INK,
                        bbox=dict(boxstyle="round,pad=0.5", fc="#eef4fb", ec=BLUE, lw=1.2))

    def state(d, s):
        return d["err"][min(s, len(d["err"]) - 1)]

    def note(d, s):
        if s == 0 or s + 1 >= d["nfock"]:
            return ""
        t = d["t"][min(s, len(d["t"]) - 1)]
        if abs(t - 1) < 0.05:
            return "edge\u2013inner charge transfer: the right amount"
        return f"edge\u2013inner charge transfer: {t:.0%} of the right amount"

    def timing(k):
        return (16, 22) if k < 6 else (8, 8)

    frames = schedule(n_steps, timing, start_hold=40, end_hold=80)

    def draw(fr):
        a, b, t = fr
        shown = b if t > 0.5 else a
        for mol, d, st, nt, nf in ((mols[0], dd, status[0], notes[0], nf_d), (mols[1], dm, status[1], notes[1], nf_m)):
            mol.update((1 - t) * state(d, a) + t * state(d, b))
            st.set_text(f"converged \u2713  {nf} Fock builds" if shown + 1 >= nf else f"Fock builds: {shown + 1}")
            nt.set_text(note(d, shown))
        for ln, hd, d, nf in ((ln_d, hd_d, dd, nf_d), (ln_m, hd_m, dm, nf_m)):
            m = min(shown, nf - 1)
            ln.set_data(np.arange(2, m + 2), d["grad"][:m])
            hd.set_data([m + 1], [d["grad"][m - 1]] if m > 0 else [np.nan])
        takeaway.set_visible(shown + 1 >= nf_d)

    names = {s: f"state{s:02d}" for s in (0, 1, 2, 3, 4, 5, nf_m - 1, n_steps)}
    render(fig, frames, draw, "acene_before_after", keyframes(frames, names))
    plt.close(fig)


def make_static(data):
    xs = np.round(data["x"], 1)
    cols = np.unique(xs)
    slice_of = np.searchsorted(cols, xs)

    def slices(err):
        return np.array([np.bincount(slice_of, weights=row, minlength=len(cols)) for row in err])

    fig = plt.figure(figsize=(12.8, 5.6))
    rows = data["diis"]["nfock"]
    for j, (m, title, col) in enumerate((("diis", "Standard DIIS", GREY), ("mk", "MK-DIIS (ours)", BLUE))):
        d = data[m]
        ax = fig.add_axes([0.07 + j * 0.25, 0.21, 0.21, 0.6])
        img = np.full((rows, len(cols)), np.nan)
        e = slices(d["err"])
        img[: len(e)] = e
        ax.imshow(img, aspect="auto", cmap=CMAP, vmin=-0.2, vmax=0.2, interpolation="nearest",
                  extent=(cols[0] - 0.6, cols[-1] + 0.6, rows + 0.5, 0.5))
        if d["nfock"] < rows:
            ax.axhspan(d["nfock"] + 0.5, rows + 0.5, color="#f3f4f6", zorder=2)
            ax.text(cols.mean(), (d["nfock"] + rows) / 2 + 0.5, "converged", ha="center", va="center", fontsize=12,
                    color=BLUE, fontweight="bold", zorder=3)
        ax.text(0, 1.085, title, transform=ax.transAxes, fontsize=13, fontweight="bold", color=col)
        ax.text(0, 1.015, f"converged after {d['nfock']} Fock builds", transform=ax.transAxes, fontsize=11,
                color=col)
        ax.set_xlabel("position along the molecule (\u00c5)")
        ax.set_yticks([1, 4, 7, 10, 13])
        if j == 0:
            ax.set_ylabel("Fock build")
        ax.text(-0.14, 1.03, "ab"[j], transform=ax.transAxes, fontsize=14, fontweight="bold", color=INK)
    cax = fig.add_axes([0.07, 0.04, 0.21, 0.025])
    cax.imshow(np.linspace(-0.2, 0.2, 256)[None, :], aspect="auto", cmap=CMAP, vmin=-0.2, vmax=0.2,
               extent=(-0.2, 0.2, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([-0.2, 0, 0.2])
    cax.set_xticklabels(["\u2264 \u22120.2", "0", "\u2265 0.2"], fontsize=9.5)
    for s in ("left", "top", "right"):
        cax.spines[s].set_visible(False)
    fig.text(0.3, 0.052, "electron error per slice across the molecule (Mulliken): blue too few, red too many.\n"
             "Columns alternate between CH groups and inner carbons.", fontsize=10.5, color=TEXT, va="center")

    ax = fig.add_axes([0.62, 0.21, 0.17, 0.6])
    for m, col, lab in (("diis", GREY, "DIIS"), ("mk", BLUE, "MK-DIIS")):
        g = data[m]["grad"]
        ax.semilogy(np.arange(2, len(g) + 2), g, "o-", color=col, ms=3.5, lw=1.6, label=lab)
    ax.axhline(np.sqrt(1e-9), color=MUTED, lw=0.9, ls=":")
    ax.set_xlabel("Fock builds")
    ax.set_ylabel("orbital gradient norm")
    ax.legend(loc="upper right", fontsize=10.5)
    ax.text(-0.3, 1.03, "c", transform=ax.transAxes, fontsize=14, fontweight="bold", color=INK)

    ax = fig.add_axes([0.86, 0.21, 0.12, 0.6])
    for m, col in (("diis", GREY), ("mk", BLUE)):
        t = data[m]["t"]
        ax.plot(np.arange(1, len(t) + 1), t, "o-", color=col, ms=3.5, lw=1.6)
    ax.axhline(1.0, color=INK, lw=0.9, ls=(0, (3, 2)))
    ax.axhline(0.0, color=MUTED, lw=0.6)
    ax.set_xlabel("Fock builds")
    ax.set_ylabel("edge\u2013inner charge transfer\n(1 = right amount)")
    ax.set_xlim(0.5, 8.5)
    ax.text(-0.45, 1.03, "d", transform=ax.transAxes, fontsize=14, fontweight="bold", color=INK)

    fig.text(0.07, 0.955, "Octacene, PBE/def2-SVP: the edge\u2013inner charge transfer sloshes under DIIS, "
             "not under MK-DIIS", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.07, 0.918, "Real PySCF runs with our code from the same MINAO guess; both reach the same energy "
             f"({data['diis']['e']:.6f} vs {data['mk']['e']:.6f} Eh).", fontsize=10.5, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "out", f"acene_static.{ext}"), dpi=150)
    print("wrote out/acene_static.{png,pdf}")


if __name__ == "__main__":
    data = load(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "trace_acene8.npz"))
    for m in ("diis", "mk"):
        print(m, data[m]["nfock"], "builds; transfer:", " ".join(f"{t:+.2f}" for t in data[m]["t"]))
    make_static(data)
    make_animation(data)
