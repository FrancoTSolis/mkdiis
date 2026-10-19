"""Figure 2 of the paper: the sloshing mode of DIIS and its absence in MK-DIIS (out/fig_sloshing.{pdf,png}).

    python paper_fig_sloshing.py [helix dir]     # helix runs default to data/

(a) toy chain (toy_chain.py): error of the charge per unit against the answer, DIIS and MK-DIIS;
(b) octacene, PBE/def2-SVP (data/trace_acene8.npz): fraction of the edge-inner charge transfer done at each build;
(c) Ace-(Ala)10-NMe, B3LYP/def2-SVP: dipole along the helix axis relative to the converged one.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import helix_figure  # noqa: E402  (imported before the rcParams below, which they would otherwise overwrite)
import real_figure  # noqa: E402
from toy_chain import SOLVERS, Chain, run

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INK, MUTED = "#111111", "#4d4d4d"
GREY, BLUE = "#8c8c8c", "#2a78d6"
plt.rcParams.update({
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4, "lines.linewidth": 1.0, "legend.frameon": False,
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.01, "pdf.fonttype": 42,
})


def panel(ax, letter, dx=-0.2):
    ax.text(dx, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def main(helix_dir):
    fig = plt.figure(figsize=(7.0, 2.3))
    ch = Chain()
    ref = ch.units(run(ch, max_cycle=300, tol=1e-11, **SOLVERS["MK-DIIS"])["charges"][-1])
    rows = len(run(ch, max_cycle=80, **SOLVERS["DIIS"])["comm"])
    for j, (m, title, col) in enumerate((("DIIS", "DIIS", MUTED), ("MK-DIIS", "MK-DIIS", BLUE))):
        ax = fig.add_axes([0.05 + j * 0.135, 0.17, 0.115, 0.68])
        err = ch.units(run(ch, max_cycle=80, **SOLVERS[m])["charges"]) - ref
        img = np.full((rows, len(ref)), np.nan)
        img[: len(err)] = err[:rows]
        im = ax.imshow(img, aspect="auto", cmap="RdBu_r", vmin=-0.01, vmax=0.01, interpolation="nearest",
                       extent=(0.5, len(ref) + 0.5, rows + 0.5, 0.5))
        ax.set_title(f"{title}, {len(err)} builds", loc="left", color=col)
        ax.set_xlabel("unit along the chain")
        ax.set_xticks([1, 15, 30])
        ax.set_yticks([1, 10, 20])
        if j == 0:
            ax.set_ylabel("Fock build")
            panel(ax, "a", -0.42)
        else:
            ax.set_yticklabels([])
    cax = fig.add_axes([0.325, 0.17, 0.008, 0.68])
    cb = fig.colorbar(im, cax=cax, ticks=[-0.01, 0, 0.01])
    cb.ax.set_yticklabels(["\u22120.01", "0", "0.01"])
    cb.ax.tick_params(labelsize=6)
    cb.ax.set_title("error (e)", fontsize=6, loc="left", pad=3)

    ac = real_figure.load(os.path.join(HERE, "data", "trace_acene8.npz"))
    ax = fig.add_axes([0.475, 0.17, 0.2, 0.68])
    for m, col, lab in (("diis", GREY, "DIIS"), ("mk", BLUE, "MK-DIIS")):
        t = ac[m]["t"]
        ax.plot(np.arange(1, len(t) + 1), t, "o-", ms=2.4, color=col, label=f"{lab} ({ac[m]['nfock']})")
    ax.axhline(1.0, color=INK, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(0.5, 10.5)
    ax.set_xlabel("Fock build")
    ax.set_ylabel("edge\u2013inner charge transfer\n(1 = converged)")
    ax.set_title("Octacene (PBE)", loc="left")
    ax.legend(loc="upper right")
    panel(ax, "b", -0.3)

    hx = helix_figure.load("ala10", helix_dir)
    ax = fig.add_axes([0.785, 0.17, 0.2, 0.68])
    for m, col, lab in (("diis", GREY, "DIIS"), ("mk_spd_cg", BLUE, "MK-DIIS")):
        mu = hx[m]["mu"]
        ax.plot(np.arange(1, len(mu) + 1), mu, "o-", ms=2.4, color=col, label=f"{lab} ({hx[m]['nfock']})")
    ax.axhline(1.0, color=INK, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(0.5, 10.5)
    ax.set_xlabel("Fock build")
    ax.set_ylabel("dipole along the helix axis\n(1 = converged)")
    ax.set_title("Ace-(Ala)$_{10}$-NMe (B3LYP)", loc="left")
    ax.legend(loc="lower right")
    panel(ax, "c", -0.3)

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    out = os.path.join(HERE, "out", "fig_sloshing")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    print("wrote out/fig_sloshing.{pdf,png}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data"))
