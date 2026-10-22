"""Figures 2 and 3 of the paper, static versions of the animations (out/fig_toy.*, out/fig_real.*).

    python paper_figs.py

Figure 2: the toy chain (toy_chain.py). (a) the chain, 60 sites drawn as 30 strongly bonded pairs ("units");
(b, c) error of the charge of each unit after each SCF step for DIIS and MK-DIIS, with the end-to-end transfer (dipole
of the chain relative to the converged one); (d) largest charge error per Fock build, including plain Roothaan steps;
(e) Fock builds against chain length.
Figure 3: Ace-(Ala)10-NMe (data/large_ala10__*.json, B3LYP/def2-SVP) after the first three SCF steps, colored by the
electron error of each residue; its gradient norm and axial dipole; octacene (data/trace_acene8.npz, PBE/def2-SVP).
"""
import os

import matplotlib
matplotlib.use("Agg")
import helix_figure  # noqa: E402  (these set rcParams on import; ours below must come after)
import real_figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from toy_chain import SOLVERS, Chain, run  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
INK, TEXT, MUTED, RULE = "#111111", "#333333", "#6e7781", "#c8ccd2"
GREY, BLUE, ORANGE = "#8c8c8c", "#2a78d6", "#d9571f"
CMAP = plt.get_cmap("RdBu_r")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "lines.linewidth": 1.0, "legend.frameon": False, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 400, "pdf.fonttype": 42,
})


def panel(letter, fig, x, y):
    fig.text(x, y, letter, fontsize=8.5, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"wrote out/{name}.{{pdf,png}}")


def colorbar(fig, rect, vmax, label, vertical=False):
    cax = fig.add_axes(rect)
    labels = [f"\u2264 \u2212{vmax:.2f}", "0", f"\u2265 {vmax:.2f}"]
    ramp = np.linspace(-vmax, vmax, 256)
    if vertical:
        cax.imshow(ramp[::-1, None], aspect="auto", cmap=CMAP, vmin=-vmax, vmax=vmax, extent=(0, 1, -vmax, vmax))
        cax.set_xticks([])
        cax.yaxis.tick_right()
        cax.set_yticks([-vmax, 0, vmax])
        cax.set_yticklabels(labels)
        for s in ("left", "top", "bottom"):
            cax.spines[s].set_visible(False)
        cax.spines["right"].set_visible(True)
    else:
        cax.imshow(ramp[None, :], aspect="auto", cmap=CMAP, vmin=-vmax, vmax=vmax, extent=(-vmax, vmax, 0, 1))
        cax.set_yticks([])
        cax.set_xticks([-vmax, 0, vmax])
        cax.set_xticklabels(labels)
        for s in ("left", "top", "right"):
            cax.spines[s].set_visible(False)
    cax.tick_params(labelsize=6, length=1.8, pad=1.5)
    cax.set_title(label, fontsize=6.3, pad=3, color=TEXT, loc="left" if vertical else "center")


def times(s):
    s = float(s)
    return f"{s:.2f}" if abs(s - 1) < 0.1 else f"{s:.1f}".replace("-", "\u2212")


def fig_toy():
    ch = Chain()
    xc = ch.x - ch.x.mean()
    ref_sites = run(ch, max_cycle=300, tol=1e-11, **SOLVERS["MK-DIIS"])["charges"][-1]
    ref = ch.units(ref_sites)
    data = {}
    for nm in ("Roothaan", "DIIS", "MK-DIIS"):
        r = run(ch, max_cycle=80, **SOLVERS[nm])
        q = ch.units(r["charges"])
        data[nm] = dict(err=q - ref, maxerr=np.abs(q - ref).max(1), sep=(r["charges"] @ xc) / (ref_sites @ xc),
                        nfock=len(r["comm"]))
    sizes = (20, 40, 60, 80, 100, 120)
    scan = {nm: [len(run(Chain(n=n), max_cycle=80, **SOLVERS[nm])["comm"]) for n in sizes] for nm in ("DIIS", "MK-DIIS")}

    fig = plt.figure(figsize=(7.0, 2.95))
    steps = [0, 1, 2, 3, 4, 5]
    labels = ["guess"] + [str(k) for k in steps[1:]]
    vmax, n = 0.03, len(ref)
    xb, wb, yb, hb = 0.072, 0.215, 0.2, 0.5

    ax = fig.add_axes([xb, 0.8, wb + 0.3, 0.13])
    xs = np.array([k + 1 + d for k in range(n) for d in (-0.19, 0.19)])
    for k in range(n - 1):
        ax.plot([xs[2 * k + 1], xs[2 * k + 2]], [0, 0], color="#b9bec6", lw=0.6, zorder=1)
    for k in range(n):
        ax.plot([xs[2 * k], xs[2 * k + 1]], [0, 0], color="#5b6370", lw=1.6, zorder=1, solid_capstyle="round")
    ax.scatter(xs, np.zeros(len(xs)), s=5, fc="white", ec="#5b6370", lw=0.5, zorder=2)
    ax.annotate("", xy=(1, 0.75), xytext=(5.5, 0.75), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.9,
                                                                        mutation_scale=7))
    ax.text(5.8, 0.75, "a weak field pulls electrons to the left", va="center", fontsize=6.3, color=ORANGE)
    ax.plot([21.81, 21.81, 22.19, 22.19], [-0.38, -0.55, -0.55, -0.38], color=TEXT, lw=0.6)
    ax.text(22.0, -0.75, "1 unit = 2 strongly bonded sites", ha="center", va="top", fontsize=6.3, color=TEXT)
    ax.text(1.0, -0.75, "60 sites = 30 units", ha="left", va="top", fontsize=6.3, color=TEXT)
    ax.set_xlim(0.4, n + 0.6)
    ax.set_ylim(-1.5, 1.2)
    ax.axis("off")
    panel("a", fig, 0.008, 0.905)

    norm = Normalize(-vmax, vmax)
    for j, (nm, col) in enumerate((("DIIS", GREY), ("MK-DIIS", BLUE))):
        d = data[nm]
        ax = fig.add_axes([xb + j * 0.3, yb, wb, hb])
        for row, k in enumerate(steps):
            ax.plot([1, n], [-row, -row], color=RULE, lw=0.5, zorder=1)
            ax.scatter(np.arange(1, n + 1), np.full(n, -row), s=13, c=CMAP(norm(d["err"][k])), ec="#5b6370",
                       lw=0.3, zorder=2)
            if j == 0:
                ax.text(-0.6, -row, labels[row], ha="right", va="center", fontsize=6.3, color=TEXT)
            if k > 0:
                ax.text(n + 1.2, -row, "\u00d7" + times(d["sep"][k]), ha="left", va="center", fontsize=6.3,
                        color=ORANGE if abs(d["sep"][k] - 1) > 0.25 else col)
        ax.set_xlim(0.4, n + 0.6)
        ax.set_ylim(-len(steps) + 0.5, 0.5)
        ax.axis("off")
        fig.text(xb + j * 0.3, yb + hb + 0.035, f"{nm}: {d['nfock']} Fock builds", ha="left", va="bottom",
                 fontsize=7, fontweight="bold", color=col)
        fig.text(xb + j * 0.3 + wb + 0.008, yb + hb + 0.012, "transfer", ha="left", va="bottom", fontsize=6,
                 color=MUTED)
        panel("bc"[j], fig, 0.008 + j * 0.3, yb + hb + 0.035)
    fig.text(0.012, yb + hb / 2, "SCF step", rotation=90, ha="center", va="center", fontsize=6.5, color=TEXT)
    colorbar(fig, [xb + 0.16, 0.075, 0.2, 0.022], vmax, "error of the charge of each unit (e)")

    ax = fig.add_axes([0.695, 0.2, 0.13, 0.7])
    for nm, col, lab in (("Roothaan", ORANGE, "plain Roothaan"), ("DIIS", GREY, "DIIS"), ("MK-DIIS", BLUE, "MK-DIIS")):
        e = data[nm]["maxerr"][:25]
        ax.semilogy(np.arange(1, len(e) + 1), e, "o-", ms=1.6, lw=0.9, color=col, label=lab)
    ax.set_xlim(0, 25)
    ax.set_ylim(1e-12, 30)
    ax.set_yticks([1e-10, 1e-7, 1e-4, 1e-1])
    ax.set_xlabel("Fock builds")
    ax.set_ylabel("largest charge error (e)")
    ax.legend(loc="lower right", handlelength=1.3, borderaxespad=0.1)
    panel("d", fig, 0.637, 0.9)

    ax = fig.add_axes([0.885, 0.2, 0.105, 0.7])
    ax.plot(sizes, scan["DIIS"], "o-", ms=2.4, color=GREY)
    ax.plot(sizes, scan["MK-DIIS"], "s-", ms=2.4, color=BLUE)
    ax.set_ylim(0, 34)
    ax.set_xticks([20, 60, 100])
    ax.set_xlabel("sites in the chain")
    ax.set_ylabel("Fock builds")
    ax.text(118, scan["DIIS"][-1] + 1.5, "DIIS", ha="right", va="bottom", fontsize=6.3, color=GREY)
    ax.text(118, scan["MK-DIIS"][-1] + 1.5, "MK-DIIS", ha="right", va="bottom", fontsize=6.3, color=BLUE)
    panel("e", fig, 0.827, 0.9)
    save(fig, "fig_toy")


def draw_helix(ax, data, err_res, clip=0.05):
    xy, sym, res = data["xy"], data["sym"], data["res"]
    heavy = np.array([s != "H" for s in sym])
    full = np.c_[xy, np.zeros(len(sym))]
    d = np.linalg.norm(full[:, None] - full[None], axis=-1)
    bonds = [(i, j) for i in range(len(sym)) for j in range(i) if heavy[i] and heavy[j] and d[i, j] < 1.7]
    ax.add_collection(LineCollection([xy[[i, j]] for i, j in bonds], colors="#c3c8cf", lw=0.5, zorder=1))
    ax.scatter(xy[heavy, 0], xy[heavy, 1], s=6.5, c=CMAP(Normalize(-clip, clip)(err_res[res[heavy]])), ec="#5b6370",
               lw=0.25, zorder=3)
    ax.set_xlim(xy[heavy, 0].min() - 0.6, xy[heavy, 0].max() + 0.6)
    ax.set_ylim(xy[heavy, 1].min() - 0.6, xy[heavy, 1].max() + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_real():
    hx = helix_figure.load("ala10")
    ac = real_figure.load(os.path.join(HERE, "data", "trace_acene8.npz"))
    fig = plt.figure(figsize=(7.0, 4.0))
    w, h, x0, top = 0.235, 0.165, 0.085, 0.705
    for i, (m, title, col) in enumerate((("diis", "DIIS", GREY), ("mk_spd_cg", "MK-DIIS", BLUE))):
        y = top - i * (h + 0.06)
        fig.text(x0 - 0.01, y + h / 2, title, ha="right", va="center", fontsize=7, fontweight="bold", color=col)
        for j, k in enumerate((1, 2, 3)):
            ax = fig.add_axes([x0 + j * (w + 0.012), y, w, h])
            draw_helix(ax, hx, hx[m]["err"][k])
            mu = hx[m]["mu"][k]
            ax.text(0.5, -0.05, f"end-to-end transfer \u00d7{times(mu)}", transform=ax.transAxes, ha="center",
                    va="top", fontsize=6.3, color=ORANGE if abs(mu - 1) > 0.25 else col)
            if i == 0:
                ax.text(0.5, 1.1, f"after SCF step {k}", transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=6.5, color=TEXT)
            if i == 1 and j == 0:
                ax.text(0.0, -0.3, "N-terminus", transform=ax.transAxes, ha="left", va="top", fontsize=5.8,
                        color=MUTED)
                ax.text(1.0, -0.3, "C-terminus", transform=ax.transAxes, ha="right", va="top", fontsize=5.8,
                        color=MUTED)
    fig.text(x0, top + h + 0.08, "Ace-(Ala)$_{10}$-NMe (B3LYP/def2-SVP), heavy atoms colored by the error of the "
             "electron count of their residue", fontsize=7, color=TEXT, va="bottom")
    panel("a", fig, 0.01, top + h + 0.08)
    colorbar(fig, [x0 + 3 * w + 0.05, top - 0.13, 0.012, 0.26], 0.05, "residue\nerror (e)", vertical=True)

    yb, hb = 0.085, 0.27
    ax = fig.add_axes([0.085, yb, 0.235, hb])
    for m, col, lab in (("diis", GREY, "DIIS"), ("mk_spd_cg", BLUE, "MK-DIIS")):
        g = hx[m]["grad"]
        ax.semilogy(np.arange(2, len(g) + 2), g, "o-", ms=1.8, lw=0.9, color=col, label=f"{lab} ({hx[m]['nfock']})")
    ax.axhline(np.sqrt(1e-9), color=MUTED, lw=0.5, ls=":")
    ax.text(10.5, np.sqrt(1e-9) * 1.6, "converged", ha="left", va="bottom", fontsize=5.8, color=MUTED)
    ax.set_xlim(0, 20.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(1e-6, 50)
    ax.set_yticks([1e-5, 1e-3, 1e-1, 10])
    ax.set_xlabel("Fock builds")
    ax.set_ylabel("orbital gradient norm")
    ax.legend(loc="upper right", handlelength=1.3, borderaxespad=0.1)
    ax.set_title("Ala$_{10}$ helix", loc="left", fontsize=7)
    panel("b", fig, 0.01, yb + hb + 0.012)

    ax = fig.add_axes([0.43, yb, 0.235, hb])
    for m, col in (("diis", GREY), ("mk_spd_cg", BLUE)):
        mu = hx[m]["mu"]
        ax.plot(np.arange(1, len(mu) + 1), mu, "o-", ms=1.8, lw=0.9, color=col)
    ax.axhline(1.0, color=INK, lw=0.5, ls=(0, (3, 2)))
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks([1, 4, 7, 10])
    ax.set_ylim(-13.5, 13.5)
    ax.set_xlabel("Fock builds")
    ax.set_ylabel("dipole along the helix axis\n(1 = converged)")
    ax.set_title("Ala$_{10}$ helix", loc="left", fontsize=7)
    panel("c", fig, 0.36, yb + hb + 0.012)

    ax = fig.add_axes([0.765, yb, 0.225, hb])
    for m, col, lab in (("diis", GREY, "DIIS"), ("mk", BLUE, "MK-DIIS")):
        t = ac[m]["t"]
        ax.plot(np.arange(1, len(t) + 1), t, "o-", ms=1.8, lw=0.9, color=col, label=f"{lab} ({ac[m]['nfock']})")
    ax.axhline(1.0, color=INK, lw=0.5, ls=(0, (3, 2)))
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks([1, 4, 7, 10])
    ax.set_ylim(-0.1, 2.15)
    ax.set_xlabel("Fock builds")
    ax.set_ylabel("edge\u2013inner charge transfer\n(1 = converged)")
    ax.set_title("octacene (PBE/def2-SVP)", loc="left", fontsize=7)
    ax.legend(loc="upper right", handlelength=1.3, borderaxespad=0.1)
    panel("d", fig, 0.697, yb + hb + 0.012)
    save(fig, "fig_real")


if __name__ == "__main__":
    fig_toy()
    fig_real()
