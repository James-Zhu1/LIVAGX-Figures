"""The big overview grid: one row per LIVAGX system, one column per PMF family.

    col 1  residue-pair PMFs (36 ordered pairs)       X6 pairs emphasised
    col 2  three-residue window PMFs (10 pairs)       AGX windows emphasised
    col 3  LIVA-LIVA fragment PMF                     mean of halves + |dPMF|/2 band
    col 4  6x6 map of pair well depths (min PMF)      shared diverging scale

Outputs (all under overall/figures/overview/):
    big_grid_mean.png            col 1 / col 4 from the mean of the two halves
    big_grid_full.png            col 1 / col 4 from the whole trajectory
    pair_well_depths_mean.csv    the numbers drawn in col 4 of big_grid_mean.png
    pair_well_depths_full.csv    the numbers drawn in col 4 of big_grid_full.png
        columns: system, i, j, pair_label, min_pmf_kcal, r_min_nm,
                 resname_pep1, resname_pep2 (Amber names; residue 6 differs between
                 the peptides in livage_cn: GLU on pep 1, GLH on pep 2),
                 tail_min (True when r_min >= 1.6 nm: no real well, value ~ 0)

PMF definition (inherited from pmf_from_cpptraj.py): -kT ln P(r) of the raw COM
distance histogram, zeroed at 1.8 nm, no r^2 Jacobian correction.

Run from overall/:
    /hopper/home/jzii2024/miniconda3/envs/pmf/bin/python scripts/fig_overview_grid.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, to_rgb  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

import style  # noqa: E402
from common import (FIGURES, POSITIONS, SYSTEMS, WINDOW_PAIRS, System,  # noqa: E402
                    assert_inside_overall, load_fragment, load_pair, load_window,
                    pair_ids, pair_label, well_depth, window_label)

OUT_DIR = "overview"
X_POS = 6  # residue position of X
X_WIN = 4  # window start of AGX

XLIM = (0.3, 1.8)  # nm
YLIM = (-1.75, 2.5)  # kcal/mol; the steep repulsive wall is clipped above
YTICKS = [-1, 0, 1, 2]
BOLD_W = 2.1  # X6-X6 / AGX-AGX emphasis
EMPH_ALPHA = 0.6
MINUS = "−"
# A pair-PMF minimum at r >= TAIL_R lies in the flat tail next to the 1.8 nm zero:
# it is noise around 0, not a contact well (all such minima are |PMF| <= 0.13).
TAIL_R = 1.6  # nm

# layout (inches)
ROW_H = 2.0
LABEL_W = 1.55
CURVE_W = 3.0
HEAT_W = 2.0
HEADER_H = 1.25
FOOTER_H = 0.45

# Amber names of positions 1-5 (identical in every system); position 6 comes from
# System.x_resname, which is "pep1/pep2" when the two peptides differ (livage_*).
AMBER_1_5 = ["LEU", "ILE", "VAL", "ALA", "GLY"]
# Charge tags for residue 6 where the LIVAGE variants differ. In livage_cn the
# prmtop has GLU on peptide 1 (res 7) and GLH on peptide 2 (res 15), so the
# per-peptide tag is what tells the X6 row and the X6 column of its map apart.
X_TAG = {"GLU": "E⁻", "GLH": "E⁰"}


def fmt(v: float, nd: int = 2) -> str:
    s = f"{v:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", MINUS)


def x_resnames(s: System) -> tuple[str, str]:
    """Amber residue-6 names on (peptide 1, peptide 2)."""
    parts = s.x_resname.split("/")
    return parts[0], parts[-1]


def resname(s: System, pep: int, pos: int) -> str:
    return AMBER_1_5[pos - 1] if pos != X_POS else x_resnames(s)[pep - 1]


def res_label(s: System, pep: int, pos: int) -> str:
    """'I2', or 'E⁻6' / 'E⁰6' for residue 6 of the LIVAGE variants."""
    tag = X_TAG.get(resname(s, pep, pos), s.sequence[pos - 1]) if pos == X_POS else s.sequence[pos - 1]
    return f"{tag}{pos}"


def pair_lab(s: System, i: int, j: int) -> str:
    """Pair label with peptide-specific residue-6 charge tag (i on pep 1, j on pep 2)."""
    return f"{res_label(s, 1, i)}–{res_label(s, 2, j)}"


# ------------------------------------------------------------------ data
def depth_table(kind: str) -> pd.DataFrame:
    rows = []
    for s in SYSTEMS:
        for i, j in pair_ids():
            d, r = well_depth(load_pair(s, i, j, kind=kind))
            rows.append(dict(system=s.key, i=i, j=j, pair_label=pair_label(i, j, s),
                             min_pmf_kcal=round(d, 4), r_min_nm=round(r, 4),
                             resname_pep1=resname(s, 1, i), resname_pep2=resname(s, 2, j),
                             tail_min=bool(r >= TAIL_R)))
    return pd.DataFrame(rows)


def depth_matrix(df: pd.DataFrame, s: System) -> tuple[np.ndarray, np.ndarray]:
    """(6x6 min PMF, 6x6 bool 'minimum lies in the tail, no well')."""
    sub = df[df.system == s.key]
    m = np.full((6, 6), np.nan)
    tail = np.zeros((6, 6), dtype=bool)
    for _, row in sub.iterrows():
        m[int(row.i) - 1, int(row.j) - 1] = row.min_pmf_kcal
        tail[int(row.i) - 1, int(row.j) - 1] = bool(row.tail_min)
    return m, tail


def row_note(s: System) -> str:
    p1, p2 = x_resnames(s)
    if p1 != p2:  # say which peptide carries which residue 6
        return f"pep 1 {p1} ({X_TAG.get(p1, s.x)})\npep 2 {p2} ({X_TAG.get(p2, s.x)})"
    return f"{s.x_resname}  ·  {s.note}"


# ------------------------------------------------------------------ panels
def setup_curve_ax(ax, bottom: bool) -> None:
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_yticks(YTICKS)
    ax.set_xticks([0.4, 0.8, 1.2, 1.6])
    ax.axhline(0, color=style.MUTED, lw=0.55, zorder=1)
    if bottom:
        ax.set_xlabel("COM distance (nm)")
    else:
        ax.tick_params(axis="x", labelbottom=False)


def corner_note(ax, lines: list[str]) -> None:
    ax.text(0.985, 0.955, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=6.6, color=style.INK_2, linespacing=1.35, zorder=6)


def draw_pairs(ax, s: System, kind: str) -> None:
    ctx, emph, bold = [], [], []
    for i, j in pair_ids():
        c = load_pair(s, i, j, kind=kind)
        n_x = (i == X_POS) + (j == X_POS)
        (ctx, emph, bold)[n_x].append(((i, j), c))
    for _, c in ctx:
        ax.plot(c.r, c.pmf, color=style.CONTEXT_GRAY, lw=style.THIN_W, zorder=2)
    for _, c in emph:
        ax.plot(c.r, c.pmf, color=s.color, lw=style.THIN_W, alpha=EMPH_ALPHA, zorder=3)
    for _, c in bold:
        ax.plot(c.r, c.pmf, color=s.color, lw=BOLD_W, zorder=4)
    allp = [(well_depth(c), ij) for ij, c in ctx + emph + bold]
    (d, r), (i, j) = min(allp, key=lambda t: t[0][0])
    (dx, _), (ix, jx) = min(((well_depth(c), ij) for ij, c in emph + bold), key=lambda t: t[0][0])
    corner_note(ax, [f"deepest {pair_lab(s, i, j)}  {fmt(d)}",
                     f"deepest X6 pair {pair_lab(s, ix, jx)}  {fmt(dx)}"])


def draw_windows(ax, s: System) -> None:
    ctx, emph, bold = [], [], []
    for a, b in WINDOW_PAIRS:
        c = load_window(s, a, b)
        n_x = (a == X_WIN) + (b == X_WIN)
        (ctx, emph, bold)[n_x].append(((a, b), c))
    for _, c in ctx:
        ax.plot(c.r, c.pmf, color=style.CONTEXT_GRAY, lw=style.THIN_W, zorder=2)
    for _, c in emph:
        ax.plot(c.r, c.pmf, color=s.color, lw=style.THIN_W, alpha=EMPH_ALPHA, zorder=3)
    for _, c in bold:
        ax.plot(c.r, c.pmf, color=s.color, lw=BOLD_W, zorder=4)
    (d, _), (a, b) = min(((well_depth(c), ab) for ab, c in ctx + emph + bold), key=lambda t: t[0][0])
    (dx, _), (ax_, bx) = min(((well_depth(c), ab) for ab, c in emph + bold), key=lambda t: t[0][0])
    corner_note(ax, [f"deepest {window_label(a, b, s)}  {fmt(d)}",
                     f"deepest AGX pair {window_label(ax_, bx, s)}  {fmt(dx)}"])


def draw_fragment(ax, s: System) -> None:
    c = load_fragment(s)
    ax.fill_between(c.r, c.pmf - c.err, c.pmf + c.err, color=s.color, alpha=0.25, lw=0, zorder=2)
    for h in (c.half1, c.half2):
        ax.plot(h[0], h[1], color=style.MUTED, lw=0.6, zorder=3)
    ax.plot(c.r, c.pmf, color=s.color, lw=style.LINE_W, zorder=4)
    d, r = well_depth(c)
    corner_note(ax, [f"min {fmt(d)} kcal/mol at {r:.2f} nm"])


def text_color(rgba) -> str:
    r, g, b = to_rgb(rgba)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if lum < 0.45 else style.INK


def draw_heat(ax, s: System, mt: tuple[np.ndarray, np.ndarray], norm: Normalize, bottom: bool):
    m, tail = mt
    im = ax.imshow(m, cmap=style.CMAP_DIV, norm=norm, origin="upper",
                   extent=(0.5, 6.5, 6.5, 0.5), interpolation="nearest")
    ax.set_anchor("W")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # rows = peptide 1, columns = peptide 2 (residue-6 tag can differ: livage_cn)
    ax.set_xticks(POSITIONS, [res_label(s, 2, k) for k in POSITIONS], fontsize=6.4)
    ax.set_yticks(POSITIONS, [res_label(s, 1, k) for k in POSITIONS], fontsize=6.4)
    ax.tick_params(length=0, pad=2)
    for i in POSITIONS:
        for j in POSITIONS:
            v = m[i - 1, j - 1]
            if not np.isfinite(v):
                continue
            if tail[i - 1, j - 1]:  # minimum in the flat tail: no real well
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=5.6,
                        color=style.MUTED, fontstyle="italic")
            else:
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=5.6,
                        color=text_color(style.CMAP_DIV(norm(v))))
    # thin white cell separators
    for k in np.arange(1.5, 6.5, 1.0):
        ax.axhline(k, color=style.SURFACE, lw=0.6)
        ax.axvline(k, color=style.SURFACE, lw=0.6)
    # outline the X6 row and X6 column (the emphasised pairs of col 1)
    for rect in (Rectangle((0.5, 5.5), 6, 1), Rectangle((5.5, 0.5), 1, 6)):
        rect.set(fill=False, ec=s.color, lw=1.0, zorder=5, clip_on=False)
        ax.add_patch(rect)
    ax.set_ylabel("peptide 1", fontsize=7, labelpad=1)
    if bottom:
        ax.set_xlabel("peptide 2", fontsize=8)
    return im


# ------------------------------------------------------------------ figure
def multicolor_line(lw: float, alpha: float = 1.0):
    return tuple(Line2D([], [], color=s.color, lw=lw, alpha=alpha, solid_capstyle="butt")
                 for s in SYSTEMS)


def build(kind: str, depths: pd.DataFrame, vmax: float):
    style.apply()
    n = len(SYSTEMS)
    widths = [LABEL_W, CURVE_W, CURVE_W, CURVE_W, HEAT_W]
    W = sum(widths) + 1.2
    H = HEADER_H + n * ROW_H + FOOTER_H
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(n, 5, width_ratios=widths, left=0.02, right=0.98,
                          top=1 - HEADER_H / H, bottom=FOOTER_H / H, wspace=0.28, hspace=0.2)
    norm = Normalize(-vmax, vmax)
    axes = np.empty((n, 5), dtype=object)
    im = None
    for r, s in enumerate(SYSTEMS):
        bottom = r == n - 1
        lab = fig.add_subplot(gs[r, 0])
        lab.axis("off")
        lab.add_patch(Rectangle((0.94, 0.08), 0.045, 0.84, transform=lab.transAxes,
                                color=s.color, lw=0, clip_on=False))
        lab.text(0.88, 0.62, s.label, transform=lab.transAxes, ha="right", va="bottom",
                 fontsize=10, color=style.INK, fontweight="bold")
        note = "\n".join(textwrap.fill(line, 24) for line in row_note(s).split("\n"))
        lab.text(0.88, 0.56, note, transform=lab.transAxes, ha="right", va="top",
                 fontsize=7, color=style.INK_2, linespacing=1.3)
        axes[r, 0] = lab

        a1 = fig.add_subplot(gs[r, 1])
        a2 = fig.add_subplot(gs[r, 2], sharex=a1, sharey=a1)
        a3 = fig.add_subplot(gs[r, 3], sharex=a1, sharey=a1)
        for a in (a1, a2, a3):
            setup_curve_ax(a, bottom)
        a1.set_ylabel(style.PMF_LABEL, fontsize=7.5)
        draw_pairs(a1, s, kind)
        draw_windows(a2, s)
        draw_fragment(a3, s)
        a4 = fig.add_subplot(gs[r, 4])
        im = draw_heat(a4, s, depth_matrix(depths, s), norm, bottom)
        axes[r, 1:] = [a1, a2, a3, a4]

    # ---- header: panel titles, per-column keys, colour bar
    src = ("mean of the two trajectory halves" if kind == "mean"
           else "full trajectory (first 5% of frames dropped)")
    halves = "mean of the two trajectory halves"
    titles = [("a   Residue-pair PMFs (36)", src),
              ("b   Three-residue windows (10)", halves + (" (no full-trajectory data)" if kind == "full" else "")),
              ("c   LIVA\u2013LIVA fragment", halves),
              ("d   Pair well-depth map",
               "min of each pair PMF; " + ("halves mean" if kind == "mean" else "full traj.")
               + f"\ngray italic = min at r \u2265 {TAIL_R:g} nm (tail, no well)")]
    top = 1 - HEADER_H / H
    title_y = top + 1.0 / H
    sub_y = top + 0.94 / H
    key_y = top + 0.1 / H
    gray = Line2D([], [], color=style.CONTEXT_GRAY, lw=style.THIN_W + 0.4)
    keys = [
        [(gray, "25 pairs without X6"),
         (multicolor_line(style.THIN_W + 0.4, EMPH_ALPHA + 0.2), "10 X6\u2013other pairs (row colour)"),
         (multicolor_line(BOLD_W), "X6\u2013X6")],
        [(gray, "6 window pairs without AGX"),
         (multicolor_line(style.THIN_W + 0.4, EMPH_ALPHA + 0.2), "3 AGX\u2013other pairs (row colour)"),
         (multicolor_line(BOLD_W), "AGX\u2013AGX")],
        [(multicolor_line(style.LINE_W), "mean of halves"),
         (tuple(Patch(color=s.color, alpha=0.25, lw=0) for s in SYSTEMS), "\u00b1 |PMF\u2081 \u2212 PMF\u2082| / 2"),
         (Line2D([], [], color=style.MUTED, lw=0.8), "first / second half")],
    ]
    for r in range(n):
        axes[r, 4].apply_aspect()  # heatmaps are square: use their real box
    lefts = [axes[0, c].get_position().x0 for c in range(1, 4)]
    hpos = axes[0, 4].get_position()
    lefts.append(hpos.x0)
    for c in range(4):
        fig.text(lefts[c], title_y, titles[c][0], ha="left", va="bottom", fontsize=10,
                 color=style.INK, fontweight="bold")
        fig.text(lefts[c], sub_y, titles[c][1], ha="left", va="top", fontsize=7.3,
                 color=style.INK_2, linespacing=1.3)
    for c, key in enumerate(keys):
        handles, labels = zip(*key)
        fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(lefts[c] - 0.004, key_y),
                   bbox_transform=fig.transFigure, ncol=1, fontsize=7.3, handlelength=2.6,
                   handletextpad=0.6, labelspacing=0.3, borderaxespad=0, borderpad=0,
                   handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
    cax = fig.add_axes([hpos.x0, top + 0.36 / H, hpos.width, 0.1 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("min PMF (kcal/mol); blue = attractive", fontsize=7.3, color=style.INK_2, labelpad=3)
    cax.xaxis.set_label_position("top")
    cb.set_ticks([-vmax, -vmax / 2, 0, vmax / 2, vmax])
    cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    cb.outline.set_visible(False)
    return fig


def caption(kind: str, vmax: float) -> str:
    lo, hi = fmt(YLIM[0]), fmt(YLIM[1], 1)
    src = ("mean of the two trajectory halves" if kind == "mean"
           else "whole trajectory after dropping the first 5% of frames (windows and fragment exist only "
                "as halves, so b and c stay mean of halves)")
    return (
        "One row per system, one column per PMF family: (a) all 36 residue-pair PMFs, gray = no X6, "
        "thin system colour = X6 with another residue, bold = X6\u2013X6; (b) the 10 three-residue window "
        "pairs, same encoding for AGX; (c) LIVA\u2013LIVA fragment, mean of halves with |PMF\u2081\u2212PMF\u2082|/2 "
        "band, halves in gray; (d) minimum of each pair PMF (rows = peptide-1 position, cols = peptide-2 "
        f"position, X6 row/column outlined; gray italic = minimum at r \u2265 {TAIL_R:g} nm, i.e. flat tail "
        "next to the reference, no well; in LIVAGE (E\u207b/E\u2070) peptide 1 is GLU and peptide 2 GLH, so "
        "its X6 row and column differ by construction). "
        f"Curve panels share x (0.3\u20131.8 nm COM distance) and y ({lo} to {hi} kcal/mol, repulsive wall "
        f"clipped) and the map uses one diverging scale \u00b1{vmax:g} kcal/mol for every row and both "
        "versions of this figure; as in pmf_from_cpptraj.py each PMF is \u2212kT ln P(r) of the raw distance "
        "histogram zeroed at 1.8 nm with no r\u00b2 Jacobian, so negative = attractive, but a non-interacting "
        "pair would sit at +2kT ln(1.8 nm/r) (\u2248 +1.3 kcal/mol at 0.6 nm) and a curve above zero is not "
        "necessarily repulsive. "
        f"Columns a and d: {src}; numbers in figures/overview/pair_well_depths_{kind}.csv.")


def main() -> None:
    out = assert_inside_overall(FIGURES / OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    tables = {k: depth_table(k) for k in ("mean", "full")}
    for k, df in tables.items():
        df.to_csv(out / f"pair_well_depths_{k}.csv", index=False)
    worst = max(np.nanmax(np.abs(df.min_pmf_kcal)) for df in tables.values())
    vmax = float(np.ceil(worst * 4) / 4)  # symmetric range shared by both figures
    for order, kind in enumerate(("mean", "full")):
        fig = build(kind, tables[kind], vmax)
        title = ("Overview grid: all PMF families × all systems"
                 + (" (mean of halves)" if kind == "mean" else " (full-trajectory pairs)"))
        p = style.save(fig, f"{OUT_DIR}/big_grid_{kind}.png", title, caption(kind, vmax),
                       "overview", [s.key for s in SYSTEMS], order, dpi=170)
        print("wrote", p, f"{p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
