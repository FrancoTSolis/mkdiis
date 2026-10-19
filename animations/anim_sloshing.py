"""Animations of charge sloshing in the 1D toy chain (toy_chain.py) and of the model-kernel correction.

    python anim_sloshing.py

writes, in out/:
    sloshing_problem.mp4             plain Roothaan steps: the charge sloshes between the ends and never settles
    sloshing_before_after.mp4        standard DIIS against MK-DIIS from the same guess, same final answer
    frames_<name>/                   key frames as PNG, and a contact sheet of each
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image

from toy_chain import SOLVERS, Chain, run

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
INK, TEXT, MUTED, RULE = "#111111", "#3f4650", "#8c959f", "#c8ccd2"
GREY, BLUE, ORANGE = "#7f7f7f", "#2a78d6", "#d9571f"
CMAP = plt.get_cmap("RdBu_r")
FPS = 20
SIZE, DPI = (12.8, 7.2), 125
X0, W = 0.09, 0.61          # left edge and width of the chain panels (figure fraction)
plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 13,
    "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.9, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "axes.labelsize": 12.5, "legend.frameon": False,
})


def toy_data():
    ch = Chain()
    xc = ch.x - ch.x.mean()
    ref_sites = run(ch, max_cycle=300, tol=1e-11, **SOLVERS["MK-DIIS"])["charges"][-1]
    ref = ch.units(ref_sites)
    data = {}
    for nm in ("Roothaan", "DIIS", "MK-DIIS"):
        r = run(ch, max_cycle=80, **SOLVERS[nm])
        q = ch.units(r["charges"])
        data[nm] = dict(q=q, err=np.abs(q - ref).max(1), sep=(r["charges"] @ xc) / (ref_sites @ xc),
                        nfock=len(r["comm"]), converged=bool(r["converged"]))
    return ch, ref, data


def ease(t):
    return t * t * (3.0 - 2.0 * t)


def schedule(n_steps, timing, start_hold, end_hold):
    """Frames as (state from, state to, blend); timing(k) -> (transition frames, hold frames)."""
    frames = [(0, 0, 0.0)] * start_hold
    for k in range(n_steps):
        nt, nh = timing(k)
        frames += [(k, k + 1, ease((j + 1) / nt)) for j in range(nt)]
        frames += [(k + 1, k + 1, 0.0)] * nh
    frames += [frames[-1]] * end_hold
    return frames


class ChainRow:
    """A strip of colored units (the molecule) above a bar chart of the charge per unit."""

    def __init__(self, fig, strip_box, bar_box, ref, ylim=0.1, vmax=0.03):
        self.n, self.ylim = len(ref), ylim
        self.norm = Normalize(-vmax, vmax)
        self.x = x = np.arange(self.n)
        ax = fig.add_axes(strip_box)
        ax.plot([0, self.n - 1], [0, 0], color=RULE, lw=2.2, zorder=1, solid_capstyle="round")
        self.circles = [Circle((i, 0), 0.4, ec="#5b6370", lw=0.7, zorder=2) for i in x]
        for c in self.circles:
            ax.add_patch(c)
        ax.set_xlim(-0.8, self.n - 0.2)
        ax.set_ylim(-0.62, 0.62)
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
        ax = fig.add_axes(bar_box)
        ax.axhline(0, color="#9aa1ab", lw=0.8, zorder=1)
        self.bars = ax.bar(x, np.zeros(self.n), width=0.8, zorder=2, lw=0.5)
        ax.step(np.r_[x[0] - 0.5, x + 0.5], np.r_[ref[0], ref], where="pre", color=INK, lw=1.6, ls=(0, (3.2, 2)),
                zorder=4)
        ax.set_xlim(-0.8, self.n - 0.2)
        ax.set_ylim(-ylim, ylim)
        ax.set_yticks([-ylim, 0, ylim])
        ax.set_yticklabels([f"{-ylim:g}", "0", f"+{ylim:g}"])
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_ylabel("charge per unit\n(electrons)", fontsize=11.5)
        self.clip_up = ax.scatter([], [], marker="^", s=46, color=INK, zorder=5, clip_on=False)
        self.clip_dn = ax.scatter([], [], marker="v", s=46, color=INK, zorder=5, clip_on=False)
        self.ax_bar = ax

    def update(self, q):
        cols = CMAP(self.norm(q))
        lim = 0.985 * self.ylim
        for c, b, v, col in zip(self.circles, self.bars, q, cols):
            c.set_facecolor(col)
            b.set_height(float(np.clip(v, -lim, lim)))
            b.set_facecolor(col)
            b.set_edgecolor("#5b6370")
        for sc, mask, y in ((self.clip_up, q > lim, 1.06), (self.clip_dn, q < -lim, -1.06)):
            sc.set_offsets(np.c_[self.x[mask], np.full(mask.sum(), y * self.ylim)] if mask.any() else np.empty((0, 2)))


def field_arrow(fig, x0, x1, y, text):
    fig.patches.append(FancyArrowPatch((x1, y), (x0, y), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=16, lw=1.6, color=ORANGE))
    fig.text(x1 + 0.008, y, text, ha="left", va="center", fontsize=12, color=ORANGE)


def legend_row(fig, y, vmax):
    """Color scale of the charge per unit and the dashed line of the self-consistent answer."""
    ax = fig.add_axes([X0, y, 0.2, 0.022])
    ax.imshow(np.linspace(-vmax, vmax, 256)[None, :], aspect="auto", cmap=CMAP, norm=Normalize(-vmax, vmax),
              extent=(-vmax, vmax, 0, 1))
    ax.set_yticks([])
    ax.set_xticks([-vmax, 0, vmax])
    ax.set_xticklabels([f"{-vmax:g}", "0", f"+{vmax:g} e"], fontsize=10)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("blue: electrons missing      red: extra electrons", fontsize=10.5, color=TEXT, pad=4)
    lax = fig.add_axes([X0 + 0.24, y - 0.01, 0.2, 0.05])
    lax.axis("off")
    lax.legend(handles=[Line2D([], [], color=INK, lw=1.6, ls=(0, (3.2, 2)), label="self-consistent answer")],
               loc="center left", fontsize=11.5, handlelength=2.6)


def render(fig, frames, draw, stem, keys):
    """Write stem.mp4; save the frames listed in keys (frame index -> name) as PNG."""
    os.makedirs(OUT, exist_ok=True)
    kdir = os.path.join(OUT, f"frames_{stem}")
    os.makedirs(kdir, exist_ok=True)
    for f in os.listdir(kdir):
        os.remove(os.path.join(kdir, f))
    mp4 = os.path.join(OUT, f"{stem}.mp4")
    writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=-1,
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow"])
    saved = []
    with writer.saving(fig, mp4, dpi=DPI):
        for i, fr in enumerate(frames):
            draw(fr)
            writer.grab_frame()
            if i in keys:
                p = os.path.join(kdir, f"{keys[i]}.png")
                fig.savefig(p, dpi=DPI)
                saved.append(p)
    contact_sheet(saved, os.path.join(kdir, "contact_sheet.png"))
    print(f"wrote {mp4} ({len(frames) / FPS:.1f} s), {len(saved)} key frames")


def contact_sheet(paths, out, cols=3, width=760):
    ims = [Image.open(p).convert("RGB") for p in paths]
    h = int(ims[0].height * width / ims[0].width)
    ims = [im.resize((width, h), Image.LANCZOS) for im in ims]
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * h), "white")
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * width, (i // cols) * h))
    sheet.save(out)


def keyframes(frames, names):
    """Index of the last frame showing each state (the end of its hold)."""
    last = {}
    for i, (a, b, t) in enumerate(frames):
        if a == b:
            last[a] = i
    return {last[s]: name for s, name in names.items() if s in last}


def times(sep):
    return f"{abs(sep):.0f}\u00d7" if abs(sep) >= 9.5 else f"{abs(sep):.1f}\u00d7"


def make_problem(ch, ref, data):
    d = data["Roothaan"]
    n_steps = 5
    q, sep, err = d["q"][: n_steps + 1], d["sep"][: n_steps + 1], d["err"][: n_steps + 1]
    captions = {
        0: "Start: neutral atoms (the usual SAD-type guess).\nThe field favours a few extra electrons on the left end.",
        1: f"Step 1: electrons move left, but {times(sep[1])} too much. The step does not know\n"
           "that charge it moves will repel the electrons that follow.",
        2: f"Step 2: that pile-up pushes back. Electrons rush to the right end:\n"
           f"{times(sep[2])} the right amount, in the wrong direction.",
        3: f"Step 3: back to the left, {times(sep[3])} too much ...",
        4: "Step 4: ... and back again. Every step overcorrects the last one.",
        5: "The plain iteration never settles: the charge sloshes from end to end.\n"
           "This is charge sloshing. Every step costs one expensive Fock build.",
    }
    fig = plt.figure(figsize=SIZE)
    fig.text(0.05, 0.935, "Charge sloshing: why SCF iterations bounce back and forth", fontsize=23,
             fontweight="bold", color=INK)
    fig.text(0.05, 0.893, "Toy model: a 60-atom conjugated chain in a weak field (like the macrodipole of a helix).  "
             "Plain SCF steps (Roothaan), no DIIS.", fontsize=13, color=MUTED)
    row = ChainRow(fig, [X0, 0.69, W, 0.1], [X0, 0.38, W, 0.28], ref)
    field_arrow(fig, X0 + 0.01, X0 + 0.15, 0.825, "weak field pulls electrons to the left")
    step_txt = fig.text(X0 + W, 0.825, "", ha="right", va="center", fontsize=15, fontweight="bold", color=INK)
    clip_txt = fig.text(X0 + W, 0.345, "", ha="right", va="center", fontsize=11.5, color=INK)
    cap = fig.text(X0, 0.29, "", ha="left", va="top", fontsize=15.5, color=INK, linespacing=1.45)
    legend_row(fig, 0.07, row.norm.vmax)

    ax = fig.add_axes([0.785, 0.38, 0.19, 0.41])
    ax.set_yscale("log")
    ax.set_xlim(0.6, n_steps + 1.4)
    ax.set_ylim(5e-3, 5)
    ax.set_xticks(range(1, n_steps + 2))
    ax.set_xlabel("Fock builds")
    ax.set_ylabel("largest charge error (e)")
    ax.set_title("the error grows", loc="left", fontsize=13, color=ORANGE, fontweight="bold")
    line, = ax.plot([], [], "o-", color=ORANGE, ms=5, lw=1.8)
    head, = ax.plot([], [], "o", color=ORANGE, ms=9)

    frames = schedule(n_steps, lambda k: (18, 44), start_hold=50, end_hold=60)

    def draw(fr):
        a, b, t = fr
        qq = (1 - t) * q[a] + t * q[b]
        row.update(qq)
        shown = b if t > 0.5 else a
        step_txt.set_text("guess" if shown == 0 else f"step {shown}")
        cap.set_text(captions[shown])
        big = np.abs(qq).max()
        clip_txt.set_text(f"\u25b2\u25bc bars cut off at \u00b1{row.ylim:g} e; actual values up to {big:.1f} e"
                          if big > row.ylim else "")
        line.set_data(np.arange(1, shown + 2), err[: shown + 1])
        head.set_data([shown + 1], [err[shown]])

    render(fig, frames, draw, "sloshing_problem", keyframes(frames, {s: f"state{s}" for s in range(n_steps + 1)}))
    plt.close(fig)


def make_before_after(ch, ref, data):
    dd, dm = data["DIIS"], data["MK-DIIS"]
    n_steps = dd["nfock"] - 1
    nf_d, nf_m = dd["nfock"], dm["nfock"]

    def state(d, s):
        return d["q"][min(s, len(d["q"]) - 1)]

    fig = plt.figure(figsize=SIZE)
    fig.text(0.05, 0.945, "Before and after the model-kernel correction", fontsize=23, fontweight="bold", color=INK)
    fig.text(0.05, 0.905, "Same toy chain, same starting guess, same final answer. Only the step differs.",
             fontsize=13, color=MUTED)
    rows, status, notes = [], [], []
    for j, (name, sub, col) in enumerate((
            ("Standard DIIS", "learns about the Coulomb push-back only after it has happened", GREY),
            ("MK-DIIS (ours)", "a cheap Coulomb model predicts the push-back before each step", BLUE))):
        top = 0.855 - j * 0.38
        fig.text(X0, top, name, fontsize=17, fontweight="bold", color=col, va="center")
        fig.text(X0, top - 0.034, sub, fontsize=12, color=MUTED, va="center")
        status.append(fig.text(X0 + W, top, "", ha="right", va="center", fontsize=14, fontweight="bold", color=col))
        row = ChainRow(fig, [X0, top - 0.125, W, 0.075], [X0, top - 0.33, W, 0.195], ref)
        notes.append(row.ax_bar.text(0.01, 0.95, "", transform=row.ax_bar.transAxes, ha="left", va="top",
                                     fontsize=12.5, fontweight="bold", color=col))
        rows.append(row)
    legend_row(fig, 0.045, rows[0].norm.vmax)

    ax = fig.add_axes([0.785, 0.22, 0.19, 0.6])
    ax.set_yscale("log")
    ax.set_xlim(0, nf_d + 2)
    ax.set_ylim(1e-8, 0.5)
    ax.set_xlabel("Fock builds")
    ax.set_ylabel("largest charge error (e)")
    ln_d, = ax.plot([], [], "o-", color=GREY, ms=3.5, lw=1.6, label="DIIS")
    ln_m, = ax.plot([], [], "o-", color=BLUE, ms=3.5, lw=1.8, label="MK-DIIS")
    hd_d, = ax.plot([], [], "o", color=GREY, ms=8)
    hd_m, = ax.plot([], [], "o", color=BLUE, ms=8)
    ax.legend(loc="upper right", fontsize=11.5, handlelength=1.6)
    end_d = ax.text(nf_d, dd["err"][-1] * 2.5, "", ha="center", va="bottom", fontsize=12, fontweight="bold",
                    color=GREY)
    end_m = ax.text(nf_m + 0.6, dm["err"][-1], "", ha="left", va="center", fontsize=12, fontweight="bold",
                    color=BLUE)
    takeaway = fig.text(0.88, 0.085, "", ha="center", va="center", fontsize=15, fontweight="bold", color=INK,
                        bbox=dict(boxstyle="round,pad=0.5", fc="#eef4fb", ec=BLUE, lw=1.2))

    def note(d, s, mk):
        if s == 0:
            return ""
        r = d["sep"][min(s, len(d["sep"]) - 1)]
        if mk:
            return "lands on the answer: no overshoot" if s < d["nfock"] else ""
        if r < -1.5:
            return f"overshoot: {times(r)} too much, wrong direction"
        if r > 1.5:
            return f"overshoot: {times(r)} too much"
        return ""

    def timing(k):
        return (16, 14) if k < 4 else ((8, 6) if k < 10 else (4, 3))

    frames = schedule(n_steps, timing, start_hold=40, end_hold=80)

    def draw(fr):
        a, b, t = fr
        shown = b if t > 0.5 else a
        for row, d, st, nt, nf, mk in ((rows[0], dd, status[0], notes[0], nf_d, False),
                                       (rows[1], dm, status[1], notes[1], nf_m, True)):
            row.update((1 - t) * state(d, a) + t * state(d, b))
            st.set_text(f"converged \u2713  {nf} Fock builds" if shown + 1 >= nf else f"Fock builds: {shown + 1}")
            nt.set_text(note(d, shown, mk))
        for ln, hd, d, nf in ((ln_d, hd_d, dd, nf_d), (ln_m, hd_m, dm, nf_m)):
            m = min(shown + 1, nf)
            ln.set_data(np.arange(1, m + 1), d["err"][:m])
            hd.set_data([m], [d["err"][m - 1]])
        end_d.set_text(f"{nf_d}" if shown + 1 >= nf_d else "")
        end_m.set_text(f"{nf_m}" if shown + 1 >= nf_m else "")
        takeaway.set_text(f"same answer,\n{nf_m} vs {nf_d} Fock builds")
        takeaway.set_visible(shown + 1 >= nf_d)

    names = {s: f"state{s:02d}" for s in (0, 1, 2, 3, 4, 6, nf_m - 1, 12, n_steps)}
    render(fig, frames, draw, "sloshing_before_after", keyframes(frames, names))
    plt.close(fig)


if __name__ == "__main__":
    ch, ref, data = toy_data()
    for nm, d in data.items():
        print(f"{nm}: {d['nfock']} Fock builds, converged {d['converged']}, separation ratios "
              + " ".join(f"{s:+.2f}" for s in d["sep"][:6]) + f", final error {d['err'][-1]:.1e}")
    make_problem(ch, ref, data)
    make_before_after(ch, ref, data)
