"""The big overview figure: every PMF, all eight LIVAGX systems overlaid.

    a  6x6 grid of residue-pair PMFs (row = pep 1 position, col = pep 2 position)
    b  4x4 upper triangle of three-residue window PMFs (LIV, IVA, VAG, AGX)
    c  LIVA-LIVA fragment PMF (in the empty lower corner of b)
    one line per system in its fixed colour; panels involving residue 6 tinted

Outputs (all under overall/figures/overview/):
    big_grid_mean.png            pair PMFs = mean of the two halves
    big_grid_full.png            pair PMFs = whole trajectory
    pair_well_depths_mean.csv    pair well depths (mean of halves)
    pair_well_depths_full.csv    pair well depths (whole trajectory)
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
GENERIC = "LIVAGX"
LINE_SYS = 1.1  # one line per system; 8 overlaid
X_TINT = "#f3f2ee"  # faint background on panels that involve residue 6 (X)
PANEL = 1.75  # inches per pair panel


def style_panel(ax, show_x: bool, show_y: bool, tint: bool = False) -> None:
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([0.4, 0.8, 1.2, 1.6])
    ax.set_yticks(YTICKS)
    ax.axhline(0, color=style.AXIS, lw=0.6, zorder=1)
    ax.tick_params(length=2, pad=1.5, labelsize=7)
    if not show_x:
        ax.tick_params(labelbottom=False)
    if not show_y:
        ax.tick_params(labelleft=False)
    if tint:
        ax.set_facecolor(X_TINT)


def overlay(ax, curves: dict) -> None:
    """One line per system, fixed system colour, in SYSTEMS order."""
    for s in SYSTEMS:
        c = curves[s.key]
        ax.plot(c.r, c.pmf, color=s.color, lw=LINE_SYS, alpha=0.9, zorder=3)


def panel_title(ax, text: str) -> None:
    ax.text(0.97, 0.94, text, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=style.INK, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc=style.SURFACE, ec="none", alpha=0.8))


def build(kind: str, depths: pd.DataFrame, vmax: float):
    pair_kind = "full" if kind == "full" else "mean"
    pairs = {(i, j): {s.key: load_pair(s, i, j, kind=pair_kind) for s in SYSTEMS} for i, j in pair_ids()}
    wins = {(a, b): {s.key: load_window(s, a, b) for s in SYSTEMS} for a, b in WINDOW_PAIRS}
    frag = {s.key: load_fragment(s) for s in SYSTEMS}

    gap = 0.9  # inches between the two blocks
    fig_w = 12 * PANEL + gap + 0.95
    fig_h = 6 * PANEL + 1.9
    fig = plt.figure(figsize=(fig_w, fig_h))
    left, bottom, top = 0.75 / fig_w, 0.55 / fig_h, 1 - 1.35 / fig_h
    w6 = 6 * PANEL / fig_w
    outer = fig.add_gridspec(1, 2, left=left, right=1 - 0.2 / fig_w, bottom=bottom, top=top,
                             width_ratios=[1, 1], wspace=gap / (6 * PANEL))
    g6 = outer[0].subgridspec(6, 6, hspace=0.09, wspace=0.09)
    g4 = outer[1].subgridspec(4, 4, hspace=0.12, wspace=0.12)  # square panels, 1.5x pair size

    # ---- a: 36 residue-pair PMFs, 8 systems each
    for i, j in pair_ids():
        ax = fig.add_subplot(g6[i - 1, j - 1])
        style_panel(ax, show_x=(i == 6), show_y=(j == 1), tint=(i == X_POS or j == X_POS))
        overlay(ax, pairs[(i, j)])
        panel_title(ax, f"{GENERIC[i - 1]}{i}–{GENERIC[j - 1]}{j}")
        if i == 1:
            ax.set_title(f"{GENERIC[j - 1]}{j}", fontsize=10, pad=4)
        if j == 1:
            ax.set_ylabel(f"{GENERIC[i - 1]}{i}\n", fontsize=10, rotation=0, ha="right", va="center", labelpad=10)
    fig.text(left + w6 / 2, top + 0.5 / fig_h, "a   Residue-pair PMFs  ·  rows: peptide 1, columns: peptide 2",
             ha="center", fontsize=12, weight="bold")

    # ---- b: 10 three-residue window PMFs (upper triangle) + c: LIVA-LIVA in the empty corner
    win_names = ["LIV", "IVA", "VAG", "AGX"]
    for a, b in WINDOW_PAIRS:
        ax = fig.add_subplot(g4[a - 1, b - 1])
        style_panel(ax, show_x=(a == b), show_y=(a == b), tint=(b == X_WIN))
        overlay(ax, wins[(a, b)])
        panel_title(ax, f"{win_names[a - 1]}–{win_names[b - 1]}")
        if a == 1:
            ax.set_title(win_names[b - 1], fontsize=10, pad=4)
        if a == b:
            ax.set_ylabel(win_names[a - 1], fontsize=10, rotation=0, ha="right", va="center", labelpad=14)
    axf = fig.add_subplot(g4[2:4, 0:2])
    style_panel(axf, show_x=True, show_y=True)
    for s in SYSTEMS:
        axf.plot(frag[s.key].r, frag[s.key].pmf, color=s.color, lw=1.7, alpha=0.9, zorder=3)
    axf.text(0.03, 0.96, "c   LIVA–LIVA fragment", transform=axf.transAxes, ha="left", va="top",
             fontsize=12, weight="bold", color=style.INK)
    axf.set_xlabel(style.R_LABEL.replace("distance", "COM distance"), fontsize=9)
    axf.set_ylabel(style.PMF_LABEL, fontsize=9)
    axf.tick_params(labelsize=8)
    for s in SYSTEMS:  # mark each system's minimum
        d, r = well_depth(frag[s.key])
        axf.plot([r], [d], "o", ms=5, color=s.color, mec=style.SURFACE, mew=1.0, zorder=4)

    x4 = outer[1].get_position(fig)
    fig.text((x4.x0 + x4.x1) / 2, top + 0.5 / fig_h, "b   Three-residue window PMFs",
             ha="center", fontsize=12, weight="bold")

    # ---- shared axis labels and legend
    fig.text(left + w6 / 2, 0.12 / fig_h, "COM distance (nm)  ·  all panels: x 0.3–1.8 nm, "
             f"y {fmt(YLIM[0])} to {fmt(YLIM[1], 1)} kcal/mol", ha="center", fontsize=9, color=style.INK_2)
    fig.text(0.12 / fig_w, bottom + (top - bottom) / 2, "PMF (kcal/mol)", rotation=90,
             va="center", fontsize=9, color=style.INK_2)
    handles = [Line2D([], [], color=s.color, lw=2.2, label=s.label) for s in SYSTEMS]
    handles.append(Patch(facecolor=X_TINT, edgecolor=style.AXIS, lw=0.5, label="panel involves residue 6 (X)"))
    fig.legend(handles=handles, loc="upper center", ncol=9, bbox_to_anchor=(0.5, 1 - 0.02 / fig_h),
               fontsize=9, handlelength=1.8, columnspacing=1.4, frameon=False)
    return fig


def caption(kind: str, vmax: float) -> str:
    src = ("mean of the two trajectory halves" if kind == "mean"
           else "whole trajectory after dropping the first 5% of frames (windows and fragment exist only "
                "as halves, so b and c stay mean of halves)")
    return (
        "Every PMF for every system in one figure, one line per system in its fixed colour. "
        "(a) the 36 residue-pair PMFs as a 6×6 grid (row = position on peptide 1, column = position on "
        "peptide 2, X = residue 6); (b) the 10 three-residue window PMFs as the upper triangle of a 4×4 grid; "
        "(c) the LIVA–LIVA fragment PMF, dots mark each system's minimum. Panels that involve residue 6 have "
        "a tinted background. All panels share x (0.3–1.8 nm COM distance) and y "
        f"({fmt(YLIM[0])} to {fmt(YLIM[1], 1)} kcal/mol, repulsive wall clipped). As in pmf_from_cpptraj.py each "
        "PMF is −kT ln P(r) of the raw distance histogram zeroed at 1.8 nm, no r² Jacobian. "
        f"Pair PMFs: {src}. Well depths are tabulated in figures/overview/pair_well_depths_{kind}.csv.")


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
        title = ("Overview: all PMFs, all systems overlaid"
                 + (" (mean of halves)" if kind == "mean" else " (full-trajectory pairs)"))
        p = style.save(fig, f"{OUT_DIR}/big_grid_{kind}.png", title, caption(kind, vmax),
                       "overview", [s.key for s in SYSTEMS], order, dpi=170)
        print("wrote", p, f"{p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
