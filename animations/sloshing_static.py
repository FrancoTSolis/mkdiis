"""Static summary of the toy-chain sloshing: space-iteration maps of the charge error, convergence, and size.

    python sloshing_static.py        # writes out/sloshing_static.{png,pdf}

Each map shows, for every Fock build (rows, top to bottom) and every unit of the chain (columns), the error of
the charge per unit against the self-consistent answer, clipped at +-0.01 e (about the answer's end charge):
alternating red and blue rows are the charge sloshing from end to end.  The convergence panel shows the tail.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toy_chain import SOLVERS, Chain, run

HERE = os.path.dirname(os.path.abspath(__file__))
INK, TEXT, MUTED = "#111111", "#3f4650", "#8c959f"
GREY, BLUE, ORANGE = "#7f7f7f", "#2a78d6", "#d9571f"
CLIP = 0.01
plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 11, "text.color": TEXT, "axes.labelcolor": TEXT,
    "axes.edgecolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.8, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.frameon": False,
})


def panel(ax, letter):
    ax.text(-0.13, 1.03, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", color=INK)


def main():
    ch = Chain()
    ref_sites = run(ch, max_cycle=300, tol=1e-11, **SOLVERS["MK-DIIS"])["charges"][-1]
    ref = ch.units(ref_sites)
    runs = {nm: run(ch, max_cycle=80, **SOLVERS[nm]) for nm in ("Roothaan", "DIIS", "MK-DIIS")}
    nrow = len(runs["DIIS"]["comm"])

    fig = plt.figure(figsize=(12.8, 6.6))
    maps = [("Roothaan", "Plain SCF step", ORANGE, 8), ("DIIS", "Standard DIIS", GREY, nrow),
            ("MK-DIIS", "MK-DIIS (ours)", BLUE, nrow)]
    for j, (nm, title, col, rows) in enumerate(maps):
        ax = fig.add_axes([0.06 + j * 0.205, 0.14, 0.17, 0.66])
        q = ch.units(runs[nm]["charges"])
        err = q - ref
        nconv = len(runs[nm]["comm"])
        img = np.full((rows, len(ref)), np.nan)
        m = min(rows, len(err))
        img[:m] = err[:m]
        ax.imshow(img, aspect="auto", cmap="RdBu_r", vmin=-CLIP, vmax=CLIP, interpolation="nearest",
                  extent=(0.5, len(ref) + 0.5, rows + 0.5, 0.5))
        if nm != "Roothaan" and nconv < rows:
            ax.axhspan(nconv + 0.5, rows + 0.5, color="#f3f4f6", zorder=2)
            ax.text(len(ref) / 2 + 0.5, (nconv + rows) / 2 + 0.5, "converged", ha="center", va="center",
                    fontsize=12, color=BLUE, fontweight="bold", zorder=3)
        ax.text(0, 1.085, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold",
                color=col)
        sub = "never settles" if nm == "Roothaan" else f"converged after {nconv} Fock builds"
        ax.text(0, 1.015, sub, transform=ax.transAxes, ha="left", va="bottom", fontsize=11, color=col)
        ax.set_xlabel("position along the chain (unit)")
        ax.set_xticks([1, 10, 20, 30])
        ax.set_yticks([1, 5, 10, 15, 20] if rows > 10 else [1, 2, 4, 6, 8])
        if j == 0:
            ax.set_ylabel("Fock build")
        panel(ax, "abc"[j])

    cax = fig.add_axes([0.06, 0.035, 0.25, 0.022])
    cax.imshow(np.linspace(-CLIP, CLIP, 512)[None, :], aspect="auto", cmap="RdBu_r", vmin=-CLIP, vmax=CLIP,
               extent=(-CLIP, CLIP, 0, 1))
    cax.set_xticks([-CLIP, -CLIP / 2, 0, CLIP / 2, CLIP])
    cax.set_xticklabels([f"\u2264 \u2212{CLIP:g}", f"\u2212{CLIP / 2:g}", "0", f"{CLIP / 2:g}", f"\u2265 {CLIP:g}"],
                        fontsize=9.5)
    cax.set_yticks([])
    for s in ("left", "top", "right"):
        cax.spines[s].set_visible(False)
    fig.text(0.325, 0.046, "error of the charge per unit (electrons): blue too few, red too many.\n"
             f"\u00b1{CLIP:g} e is about the charge on each end unit in the answer.", fontsize=10.5, color=TEXT,
             va="center")

    ax = fig.add_axes([0.71, 0.56, 0.27, 0.28])
    for nm, col, lab in (("Roothaan", ORANGE, "plain step"), ("DIIS", GREY, "DIIS"), ("MK-DIIS", BLUE, "MK-DIIS")):
        e = np.abs(ch.units(runs[nm]["charges"]) - ref).max(1)[:nrow]
        ax.semilogy(np.arange(1, len(e) + 1), e, "o-", ms=2.8, lw=1.4, color=col, label=lab)
    ax.set_ylim(1e-8, 5)
    ax.set_xlim(0, nrow + 1)
    ax.set_xlabel("Fock builds")
    ax.set_ylabel("largest charge error (e)")
    ax.legend(loc="center right", fontsize=10)
    panel(ax, "d")

    ax = fig.add_axes([0.71, 0.14, 0.27, 0.28])
    ns = np.array([20, 40, 60, 80, 100, 120])
    nd, nm_ = [], []
    for n in ns:
        c = Chain(n=n)
        nd.append(len(run(c, max_cycle=80, **SOLVERS["DIIS"])["comm"]))
        nm_.append(len(run(c, max_cycle=80, **SOLVERS["MK-DIIS"])["comm"]))
    ax.plot(ns, nd, "o-", color=GREY, ms=4, lw=1.6, label="DIIS")
    ax.plot(ns, nm_, "s-", color=BLUE, ms=4, lw=1.8, label="MK-DIIS")
    ax.set_ylim(0, max(nd) + 4)
    ax.set_xticks(ns)
    ax.set_xlabel("atoms in the chain")
    ax.set_ylabel("Fock builds")
    ax.legend(loc="upper left", fontsize=10)
    panel(ax, "e")

    fig.text(0.06, 0.955, "Charge sloshing in a 60-atom toy chain: plain step, DIIS and MK-DIIS from the same guess",
             fontsize=16, fontweight="bold", color=INK)
    fig.text(0.06, 0.918, "Rows that alternate between red and blue are charge moving from one end to the other and "
             "back.  Toy model: tight-binding chain with a Hartree + local xc mean field in a weak field.",
             fontsize=10.5, color=MUTED)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "out", f"sloshing_static.{ext}"), dpi=150)
    print("wrote out/sloshing_static.{png,pdf}; DIIS vs MK-DIIS by chain length:", list(zip(ns, nd, nm_)))


if __name__ == "__main__":
    main()
