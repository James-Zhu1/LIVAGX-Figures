#!/usr/bin/env python3
"""Cross-system PMF comparison: the same PMF overlaid for all eight LIVAGX systems.

Run from overall/:

    /hopper/home/jzii2024/miniconda3/envs/pmf/bin/python scripts/fig_cross_system.py

Outputs (all under overall/figures/cross_system/, registered in figures/index.json
with group 'cross_system'):

    pairs_6x6_mean.png          36 residue-pair PMFs, mean of the two trajectory halves
    pairs_6x6_full.png          36 residue-pair PMFs, whole trajectory
    windows_mean.png            10 three-residue window-pair PMFs (upper triangle)
    fragment_mean.png           LIVA-LIVA fragment: overlay + per-system halves/band
    x_residue_focus.png         every PMF that involves residue 6 (X)
    livage_protonation.png      LIVAGE charge states only (E-/E-, E-/E0, E0/E0)
    well_depth_heatmap.png      min of the mean PMF, 8 systems x 47 PMFs
    well_position_heatmap.png   distance of that minimum
    well_depth_vs_livagg.png    well depth minus the LIVAGG (X = Gly) well depth
    well_depths_all.csv         tidy table behind the heatmaps

Conventions (identical to pmf_from_cpptraj.py): pair PMFs 0.01 nm bins, window and
fragment PMFs 0.02 nm bins; every curve is zero at the bin nearest 1.8 nm; "mean" =
average of the two 5%-trimmed trajectory halves on a common grid, error =
|PMF1 - PMF2| / 2. Colours: one fixed hue per system from common.SYSTEMS.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402
import style as S  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (backend already set by style)
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

GROUP = "cross_system"
OUTDIR = "cross_system"

PAIR_XLIM = (0.3, C.NORMALIZATION_DISTANCE)
GROUP_XLIM = (0.2, C.NORMALIZATION_DISTANCE)
YLIM = (-1.7, 1.5)
YTICKS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
X_YLIM = (-1.0, 1.25)  # X-involving PMFs are shallow: zoom (still shared within the figure)
E_YLIM = (-1.5, 1.25)  # LIVAGE figure also shows the deeper LIVA-LIVA fragment
XTICKS = [0.4, 0.8, 1.2, 1.6]
NO_WELL = -0.1  # kcal/mol: a minimum shallower than this is "no defined well"
TAIL_R = 1.6  # nm: a minimum at r >= this sits in the flat tail next to the 1.8 nm zero
#               (same rule as fig_overview_grid.py), so it is not a contact well either
X_TINT = "#f3f2ee"  # panel background for PMFs that involve residue 6
GEN = "LIVAGX"

SYSTEMS = C.SYSTEMS
LIVAGE = [C.BY_KEY[k] for k in ("livage_cc", "livage_cn", "livage_nn")]
REF_KEY = "livagg"

ASYM_NOTE = ("In LIVAGE (E⁻/E⁰) Glu6 of peptide 1 is charged (GLU) and Glu6 of "
             "peptide 2 is neutral (GLH), so L1–X6 and X6–L1 differ by charge there.")


# ------------------------------------------------------------------ data
class Store:
    """Load every curve once."""

    def __init__(self) -> None:
        self.pair: dict[tuple[str, str, int, int], C.Curve] = {}
        self.window: dict[tuple[str, int, int], C.Curve] = {}
        self.fragment: dict[str, C.Curve] = {}
        for s in SYSTEMS:
            for i, j in C.pair_ids():
                for kind in ("mean", "full"):
                    self.pair[(s.key, kind, i, j)] = C.load_pair(s, i, j, kind)
            for a, b in C.WINDOW_PAIRS:
                self.window[(s.key, a, b)] = C.load_window(s, a, b)
            self.fragment[s.key] = C.load_fragment(s)

    def get(self, s: C.System, family: str, ident, kind: str = "mean") -> C.Curve:
        if family == "pair":
            return self.pair[(s.key, kind, *ident)]
        if family == "window":
            return self.window[(s.key, *ident)]
        return self.fragment[s.key]


def all_pmf_ids() -> list[tuple[str, object, str]]:
    """(family, id, generic label) for the 47 PMFs in display order."""
    out = [("pair", (i, j), C.pair_label(i, j)) for i, j in C.pair_ids()]
    out += [("window", (a, b), C.window_label(a, b)) for a, b in C.WINDOW_PAIRS]
    out += [("fragment", None, "LIVA–LIVA")]
    return out


def specific_label(s: C.System, family: str, ident) -> str:
    if family == "pair":
        return C.pair_label(*ident, system=s)
    if family == "window":
        return C.window_label(*ident, system=s)
    return "LIVA–LIVA"


def involves_x(family: str, ident) -> bool:
    if family == "pair":
        return 6 in ident
    if family == "window":
        return 4 in ident  # window starting at 4 = AGX
    return False


def num(v: float, signed: bool = False) -> str:
    """2-decimal label with a real minus sign and no '-0.00'."""
    s = f"{v:+.2f}" if signed else f"{v:.2f}"
    if s.lstrip("+-") == "0.00":
        s = "0.00"
    return s.replace("-", "\u2212")


def err_at(c: C.Curve, k: int) -> float:
    return float(c.err[k]) if c.err is not None and k < len(c.err) else float("nan")


def well_table(store: Store) -> pd.DataFrame:
    rows = []
    for s in SYSTEMS:
        for family, ident, glabel in all_pmf_ids():
            kinds = ("mean", "full") if family == "pair" else ("mean",)
            for kind in kinds:
                c = store.get(s, family, ident, kind)
                depth, rmin = C.well_depth(c)
                k = int(np.nanargmin(c.pmf)) if np.isfinite(c.pmf).any() else -1
                err = err_at(c, k) if (kind == "mean" and k >= 0) else float("nan")
                rows.append(dict(
                    system=s.key, family=family, kind=kind,
                    pmf_id=glabel.replace("–", "-"),
                    min_pmf_kcal=round(depth, 4), r_min_nm=round(rmin, 4),
                    err_at_min_kcal=round(err, 4) if np.isfinite(err) else np.nan,
                    pmf_id_specific=specific_label(s, family, ident).replace("–", "-"),
                    system_label=s.label, sequence=s.sequence, x_resname=s.x_resname,
                ))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ helpers
def system_handles(systems, lw=2.0):
    return [Line2D([], [], color=s.color, lw=lw, label=s.label) for s in systems]


def pmf_axis(ax, xlim, tint=False, ylim=YLIM):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(XTICKS)
    ax.set_yticks([v for v in YTICKS if ylim[0] <= v <= ylim[1]])
    ax.axhline(0.0, color=S.AXIS, lw=0.6, zorder=1)
    if tint:
        ax.set_facecolor(X_TINT)


def draw_systems(ax, store, systems, family, ident, kind="mean", lw=S.THIN_W, band=False):
    for s in systems:
        c = store.get(s, family, ident, kind)
        if band and c.err is not None:
            ax.fill_between(c.r, c.pmf - c.err, c.pmf + c.err, color=s.color,
                            alpha=0.18, lw=0, zorder=2)
        ax.plot(c.r, c.pmf, color=s.color, lw=lw, zorder=3, label=s.label)


def corner_label(ax, text, size=8.5):
    ax.text(0.96, 0.95, text, transform=ax.transAxes, ha="right", va="top",
            fontsize=size, color=S.INK)


def legend_in_slot(fig, axes_list, handles, ncol=2, fontsize=9, title=None):
    """Put a legend centred on the figure area spanned by (hidden) axes."""
    x0 = min(a.get_position().x0 for a in axes_list)
    x1 = max(a.get_position().x1 for a in axes_list)
    y0 = min(a.get_position().y0 for a in axes_list)
    y1 = max(a.get_position().y1 for a in axes_list)
    leg = fig.legend(handles=handles, loc="center", bbox_to_anchor=((x0 + x1) / 2, (y0 + y1) / 2),
                     ncol=ncol, fontsize=fontsize, handlelength=2.2, columnspacing=1.6,
                     labelspacing=0.7, title=title, title_fontsize=fontsize)
    if title:
        leg.get_title().set_color(S.INK_2)
    return leg


# ------------------------------------------------------------------ (a)/(b) 6x6 pairs
def fig_pairs(store: Store, kind: str, order: int):
    fig, axes = plt.subplots(6, 6, figsize=(15, 14.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.055, top=0.895, wspace=0.07, hspace=0.09)
    for r, i in enumerate(C.POSITIONS):
        for c, j in enumerate(C.POSITIONS):
            ax = axes[r, c]
            pmf_axis(ax, PAIR_XLIM, tint=involves_x("pair", (i, j)))
            draw_systems(ax, store, SYSTEMS, "pair", (i, j), kind, lw=0.9)
            corner_label(ax, C.pair_label(i, j))
        axes[0, r].set_title(f"{GEN[r]}{r + 1}", fontsize=11, pad=6)
        axes[r, 0].annotate(f"{GEN[r]}{r + 1}", xy=(-0.36, 0.5), xycoords="axes fraction",
                            rotation=0, ha="center", va="center", fontsize=11, color=S.INK)
    fig.text(0.54, 0.915, "column = residue on peptide 2", ha="center", va="bottom", fontsize=10,
             color=S.INK_2)
    fig.text(0.008, 0.475, "row = residue on peptide 1", ha="left", va="center", rotation=90,
             fontsize=10, color=S.INK_2)
    fig.supxlabel(S.R_LABEL, fontsize=10, y=0.02)
    fig.text(0.045, 0.475, S.PMF_LABEL, ha="center", va="center", rotation=90, fontsize=10, color=S.INK_2)
    what = ("mean of the two trajectory halves" if kind == "mean"
            else "full trajectory, first 5% dropped")
    fig.suptitle(f"Residue-pair PMFs across the eight LIVAGX systems ({what})", fontsize=13, y=0.985)
    fig.legend(handles=system_handles(SYSTEMS), loc="upper center", bbox_to_anchor=(0.54, 0.965),
               ncol=8, fontsize=9.5, handlelength=2.2, columnspacing=1.8)
    if kind == "mean":
        cap = ("Each panel overlays the eight systems' PMF of the centre-of-mass distance between residue i of "
               "peptide 1 (row) and residue j of peptide 2 (column); X = residue 6 (E, G, H, K, Q or R) and the "
               "tinted panels are the 11 pairs that involve it. Curves are the mean of the two 5%-trimmed trajectory "
               "halves (single_residue_pmf_mean_value, 0.01 nm bins, zero at 1.8 nm), so a negative dip is an "
               "attractive contact well; all panels share axes. The half-to-half error at each minimum is in "
               "well_depths_all.csv (bands: livage_protonation.png and each system's own 6x6_mean_uncertainty plot).")
        title = "Residue-pair PMFs, all systems (mean of halves)"
    else:
        cap = ("Same 6x6 layout as the mean-of-halves grid, but each curve is the PMF of the full trajectory with "
               "the first 5% of frames dropped as equilibration (single_residue_pmf_full_value, 0.01 nm bins, zero "
               "at 1.8 nm, no uncertainty estimate). Rows = residue on peptide 1, columns = residue on peptide 2, "
               "X = residue 6, tinted panels involve X; shared axes.")
        title = "Residue-pair PMFs, all systems (full trajectory)"
    return S.save(fig, f"{OUTDIR}/pairs_6x6_{kind}.png", title, cap, GROUP, order=order, dpi=150)


# ------------------------------------------------------------------ (c) windows
def fig_windows(store: Store, order: int):
    n = len(C.WINDOW_STARTS)
    fig, axes = plt.subplots(n, n, figsize=(12, 11.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.07, top=0.92, wspace=0.08, hspace=0.1)
    for r, a in enumerate(C.WINDOW_STARTS):
        for c, b in enumerate(C.WINDOW_STARTS):
            ax = axes[r, c]
            if b < a:
                ax.set_visible(False)
                continue
            pmf_axis(ax, GROUP_XLIM, tint=involves_x("window", (a, b)))
            draw_systems(ax, store, SYSTEMS, "window", (a, b), "mean", lw=1.0)
            corner_label(ax, C.window_label(a, b), size=9)
            if r == c:  # nothing below / left of a diagonal panel: label it
                ax.tick_params(labelleft=True, labelbottom=True)
                ax.set_xlabel(S.R_LABEL)
                ax.set_ylabel(S.PMF_LABEL)
        axes[0, r].set_title(C.window_name(None, a), fontsize=11, pad=6)
    # legend in the empty lower triangle
    legend_in_slot(fig, [axes[2, 0], axes[3, 0], axes[3, 1]], system_handles(SYSTEMS), ncol=1,
                   fontsize=10, title="system")
    fig.suptitle("Three-residue window PMFs across the eight LIVAGX systems (mean of halves)",
                 fontsize=13, y=0.98)
    cap = ("Upper triangle of the 4x4 grid of three-residue window pairs (LIV, IVA, VAG, AGX; X = residue 6); "
           "off-diagonal panels pool both orderings (window i on peptide 1 with j on peptide 2 and vice versa), as "
           "in pmf_from_cpptraj.py, so LIV–AGX is the same curve as AGX–LIV. Curves are the mean of the two "
           "trajectory halves (three_residue_pmf_halves_value, 0.02 nm bins, zero at 1.8 nm); tinted panels contain "
           "the AGX window; shared axes.")
    return S.save(fig, f"{OUTDIR}/windows_mean.png", "Three-residue window PMFs, all systems",
                  cap, GROUP, order=order, dpi=160)


# ------------------------------------------------------------------ (d) fragment
def fig_fragment(store: Store, order: int):
    fig = plt.figure(figsize=(16.5, 6.6))
    gs = fig.add_gridspec(2, 6, width_ratios=[2.5, 0.28, 1, 1, 1, 1], left=0.05, right=0.995,
                          bottom=0.1, top=0.8, wspace=0.1, hspace=0.3)
    big = fig.add_subplot(gs[:, 0])
    pmf_axis(big, GROUP_XLIM)
    draw_systems(big, store, SYSTEMS, "fragment", None, lw=S.LINE_W)
    big.set_xlabel(S.R_LABEL)
    big.set_ylabel(S.PMF_LABEL)
    big.set_title("all systems (mean of halves)", fontsize=10.5, loc="left")
    smalls = []
    for k, s in enumerate(SYSTEMS):
        r, c = divmod(k, 4)
        ax = fig.add_subplot(gs[r, 2 + c], sharex=big, sharey=big)
        smalls.append(ax)
        pmf_axis(ax, GROUP_XLIM)
        cv = store.get(s, "fragment", None)
        (r1, p1), (r2, p2) = cv.half1, cv.half2
        ax.plot(r1, p1, color=S.CONTEXT_GRAY, lw=S.THIN_W, zorder=3.5)
        ax.plot(r2, p2, color=S.MUTED, lw=S.THIN_W, zorder=3.5)
        ax.fill_between(cv.r, cv.pmf - cv.err, cv.pmf + cv.err, color=s.color, alpha=0.25, lw=0, zorder=3)
        ax.plot(cv.r, cv.pmf, color=s.color, lw=S.LINE_W, zorder=4)
        kmin = int(np.nanargmin(cv.pmf))
        ax.text(0.96, 0.05, f"min {num(cv.pmf[kmin])} ± {num(cv.err[kmin])} kcal/mol\nat {cv.r[kmin]:.2f} nm",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=S.INK_2)
        ax.set_title(s.label, fontsize=9.5, loc="left")
        if c != 0:
            ax.tick_params(labelleft=False)
        if r == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(S.R_LABEL, fontsize=8.5)
    fig.legend(handles=system_handles(SYSTEMS), loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=8, fontsize=9.5, handlelength=2.2, columnspacing=1.8)
    enc = [Line2D([], [], color=S.INK_2, lw=S.LINE_W, label="mean of halves (system colour)"),
           Patch(facecolor=S.INK_2, alpha=0.25, lw=0, label="± |PMF₁ − PMF₂| / 2"),
           Line2D([], [], color=S.CONTEXT_GRAY, lw=S.THIN_W * 1.5, label="1st half"),
           Line2D([], [], color=S.MUTED, lw=S.THIN_W * 1.5, label="2nd half")]
    x0 = smalls[0].get_position().x0
    x1 = smalls[3].get_position().x1
    fig.legend(handles=enc, loc="lower center", bbox_to_anchor=((x0 + x1) / 2, 0.835), ncol=4,
               fontsize=8.5, handlelength=2.0, columnspacing=1.5)
    fig.suptitle("LIVA–LIVA fragment PMF across the eight LIVAGX systems", fontsize=13, y=1.0)
    cap = ("PMF of the centre-of-mass distance between the LIVA fragments (positions 1-4 = residues 2-5 and "
           "10-13, so residue 6 itself is not included) of the two peptides: left, the eight "
           "systems' mean-of-halves PMFs overlaid; right, one panel per system with the two 5%-trimmed trajectory "
           "halves in gray, the mean in the system colour, the ±|PMF₁−PMF₂|/2 band and the well minimum ± its "
           "half-to-half error. Data: fragment_pmf_halves_value/LIVA_LIVA_halves.csv (0.02 nm bins, zero at "
           "1.8 nm); shared axes.")
    return S.save(fig, f"{OUTDIR}/fragment_mean.png", "LIVA–LIVA fragment PMF, all systems",
                  cap, GROUP, order=order, dpi=160)


# ------------------------------------------------------------------ (e)/(f) X focus
def focus_layout(store, systems, top_items, row_labels, band, lw, title, figsize, label_sys=None,
                 ylim=YLIM):
    """3 x 5 grid: row 0 = headline PMFs (+ legend slot), row 1 = residue i (pep 1) vs X6
    (pep 2), row 2 = X6 (pep 1) vs residue j (pep 2)."""
    fig, axes = plt.subplots(3, 5, figsize=figsize, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.07, top=0.9, wspace=0.08, hspace=0.3)
    lab_sys = label_sys  # None -> generic LIVAGX labels

    def label(family, ident):
        if family == "fragment":
            return "LIVA–LIVA (fragment)"
        if family == "window":
            return f"{C.window_label(*ident, system=lab_sys)} (window)"
        return C.pair_label(*ident, system=lab_sys)

    for c, (family, ident) in enumerate(top_items):
        ax = axes[0, c]
        pmf_axis(ax, GROUP_XLIM, ylim=ylim)
        draw_systems(ax, store, systems, family, ident, lw=lw, band=band)
        ax.set_title(label(family, ident), fontsize=10)
    for c in range(len(top_items), 5):
        axes[0, c].set_visible(False)
    for c, i in enumerate(range(1, 6)):
        for r, ident in ((1, (i, 6)), (2, (6, i))):
            ax = axes[r, c]
            pmf_axis(ax, GROUP_XLIM, ylim=ylim)
            draw_systems(ax, store, systems, "pair", ident, lw=lw, band=band)
            ax.set_title(label("pair", ident), fontsize=10)
    # the hidden top-row slots: re-enable x tick labels on the panels above them
    for c in range(len(top_items)):
        axes[0, c].tick_params(labelbottom=True)
    for r in range(3):
        axes[r, 0].annotate(row_labels[r], xy=(-0.42, 0.5), xycoords="axes fraction", rotation=90,
                            ha="center", va="center", fontsize=9.5, color=S.INK)
    fig.supxlabel(S.R_LABEL, fontsize=10, y=0.015)
    fig.text(0.04, 0.48, S.PMF_LABEL, ha="center", va="center", rotation=90, fontsize=10, color=S.INK_2)
    handles = system_handles(systems)
    if band:
        handles.append(Patch(facecolor=S.INK_2, alpha=0.18, lw=0, label="± |PMF₁ − PMF₂| / 2 (halves)"))
    slots = [axes[0, c] for c in range(len(top_items), 5)]
    legend_in_slot(fig, slots, handles, ncol=1 if len(slots) == 1 else 2, fontsize=9.5,
                   title="system")
    fig.suptitle(title, fontsize=13, y=0.975)
    return fig


def fig_x_focus(store: Store, order: int):
    top = [("pair", (6, 6)), ("window", (4, 4)), ("window", (1, 4))]
    rows = ["X6 self-contact\n& AGX windows", "peptide 1 residue\nvs X6 of peptide 2",
            "X6 of peptide 1\nvs peptide 2 residue"]
    fig = focus_layout(store, SYSTEMS, top, rows, band=False, lw=1.1,
                       title="How the sixth residue X changes association: every PMF that involves X6",
                       figsize=(16, 10.2), ylim=X_YLIM)
    cap = ("Every PMF that involves the varying residue 6 (X): X6–X6 and the AGX–AGX and LIV–AGX windows (top), "
           "the other residue pairs with X on peptide 2 (middle) or on peptide 1 (bottom); one line per system, "
           "mean of the two trajectory halves, zero at 1.8 nm, shared y-axis zoomed to −1.0…1.25 kcal/mol because "
           "X-involving wells are shallow. More negative = stronger association: X6–X6 has no attractive well in "
           "any system (most repulsive for E⁻/E⁻, K and H⁺) and LIVAGH (H⁺) gives the shallowest wells almost "
           "everywhere. " + ASYM_NOTE)
    return S.save(fig, f"{OUTDIR}/x_residue_focus.png", "Residue-6 (X) focus: PMFs involving X6",
                  cap, GROUP, order=order, dpi=160)


def fig_livage(store: Store, order: int):
    top = [("pair", (6, 6)), ("window", (4, 4)), ("window", (1, 4)), ("fragment", None)]
    rows = ["E6 self-contact,\nAGE windows, fragment", "peptide 1 residue vs E6\nof peptide 2 (E⁰ in E⁻/E⁰)",
            "E6 of peptide 1 (E⁻ in E⁻/E⁰)\nvs peptide 2 residue"]
    fig = focus_layout(store, LIVAGE, top, rows, band=True, lw=S.LINE_W,
                       title="LIVAGE protonation states: Glu6 charged/neutral on each peptide",
                       figsize=(16, 10.2), label_sys=LIVAGE[0], ylim=E_YLIM)
    cap = ("Only the three LIVAGE charge states: both Glu6 charged (E⁻/E⁻, GLU/GLU), peptide 1 charged and "
           "peptide 2 neutral (E⁻/E⁰, GLU/GLH), both neutral (E⁰/E⁰, GLH/GLH). Top: E6–E6, the AGE–AGE and LIV–AGE "
           "windows and the LIVA–LIVA fragment; middle/bottom: the other E6-involving pairs with E6 on peptide 2 "
           "(neutral in E⁻/E⁰) or on peptide 1 (charged in E⁻/E⁰); the LIV–AGE window pools both orderings, so in "
           "E⁻/E⁰ it mixes the charged and the neutral AGE. Lines = mean of the two trajectory halves, "
           "bands = ±|PMF₁−PMF₂|/2, zero at 1.8 nm, shared y-axis zoomed to −1.5…1.25 kcal/mol.")
    return S.save(fig, f"{OUTDIR}/livage_protonation.png", "LIVAGE protonation states compared",
                  cap, GROUP, systems=[s.key for s in LIVAGE], order=order, dpi=160)


# ------------------------------------------------------------------ (g) heatmaps
CELL = 0.36  # inch
GAP = 0.32  # inch between PMF families


def heatmap_figure(matrix, rows, cmap, norm, fmt, cbar_label, title, text_dark=None, blank=None,
                   muted=None, note=None):
    """matrix: (n_rows, 47). Columns split into pairs | windows | fragment axes.
    muted: bool mask of cells to wash out (fill faded toward the surface, gray value)."""
    ids = all_pmf_ids()
    splits = [(0, 36, "single-residue pairs  (residue on peptide 1 – residue on peptide 2)"),
              (36, 46, "three-residue windows"), (46, 47, "fragment")]
    n = len(rows)
    left, bottom, top_m = 2.0, 0.95, 0.95
    width = left + 47 * CELL + 2 * GAP + 0.5 + 0.18 + 0.9
    height = bottom + n * CELL + top_m
    fig = plt.figure(figsize=(width, height))
    x = left
    axes = []
    im = None
    for k, (a, b, header) in enumerate(splits):
        w = (b - a) * CELL
        ax = fig.add_axes([x / width, bottom / height, w / width, n * CELL / height])
        axes.append(ax)
        sub = matrix[:, a:b]
        shown = np.ma.masked_invalid(sub if blank is None else np.where(blank[:, a:b], np.nan, sub))
        im = ax.imshow(shown, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_facecolor(S.SURFACE)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(range(b - a))
        labels = [ids[q][2] for q in range(a, b)]
        ax.set_xticklabels(labels, rotation=90, fontsize=7.5)
        for t, q in zip(ax.get_xticklabels(), range(a, b)):
            if involves_x(ids[q][0], ids[q][1]):
                t.set_fontweight("bold")
                t.set_color(S.INK)
        ax.tick_params(length=0, pad=3)
        if k == 0:
            ax.set_yticks(range(n))
            ax.set_yticklabels([s.label for s in rows], fontsize=9)
            ax.tick_params(axis="y", pad=19)
            for rr, s in enumerate(rows):
                ax.plot([-0.95], [rr], marker="s", ms=7, color=s.color, clip_on=False, zorder=5,
                        scalex=False, scaley=False)
            for v in range(6, 36, 6):
                ax.axvline(v - 0.5, color=S.SURFACE, lw=2.5)
            for blk in range(6):
                ax.text(blk * 6 + 2.5, -0.75, f"peptide 1: {GEN[blk]}{blk + 1}", ha="center", va="bottom",
                        fontsize=8, color=S.INK_2)
        else:
            ax.set_yticks([])
        for rr in range(1, n):
            ax.axhline(rr - 0.5, color=S.SURFACE, lw=1.0)
        ax.text((b - a - 1) / 2, -0.75 - (0.75 if k == 0 else 0.0), header, ha="center", va="bottom",
                fontsize=9.5 if k < 2 else 8.5, color=S.INK)
        # annotations
        for rr in range(n):
            for cc in range(b - a):
                v = matrix[rr, a + cc]
                if blank is not None and blank[rr, a + cc]:
                    ax.text(cc, rr, "–", ha="center", va="center", fontsize=7, color=S.MUTED)
                    continue
                if not np.isfinite(v):
                    continue
                col = "white" if (text_dark is not None and text_dark(v)) else S.INK
                if muted is not None and muted[rr, a + cc]:
                    # within error: fade the fill so the colour does not over-state it
                    ax.add_patch(Rectangle((cc - 0.5, rr - 0.5), 1, 1, facecolor=S.SURFACE, alpha=0.6,
                                           lw=0, zorder=1.5))
                    col = S.MUTED
                ax.text(cc, rr, fmt(v), ha="center", va="center", fontsize=6.4, color=col)
        ax.set_xlim(-0.5, b - a - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        x += w + GAP
    cax = fig.add_axes([(x - GAP + 0.5) / width, bottom / height, 0.18 / width, n * CELL / height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label, fontsize=9, color=S.INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8)
    if note:
        fig.text((x - GAP + 0.59) / width, (bottom - 0.12) / height, note, ha="center", va="top",
                 fontsize=7.5, color=S.INK_2, linespacing=1.3)
    fig.suptitle(title, fontsize=12.5, y=1 - 0.12 / height, x=(left + 47 * CELL / 2) / width)
    return fig


def well_matrices(df: pd.DataFrame):
    m = df[df.kind == "mean"]
    ids = [g.replace("–", "-") for _, _, g in all_pmf_ids()]
    depth = np.full((len(SYSTEMS), len(ids)), np.nan)
    rmin = depth.copy()
    err = depth.copy()
    for r, s in enumerate(SYSTEMS):
        sub = m[m.system == s.key].set_index("pmf_id")
        depth[r] = sub.loc[ids, "min_pmf_kcal"].to_numpy(float)
        rmin[r] = sub.loc[ids, "r_min_nm"].to_numpy(float)
        err[r] = sub.loc[ids, "err_at_min_kcal"].to_numpy(float)
    return depth, rmin, err


def fig_well_depth(df: pd.DataFrame, order: int):
    depth, _, _ = well_matrices(df)
    vmax = float(np.ceil(np.nanmax(np.abs(depth)) * 10) / 10)
    norm = Normalize(-vmax, vmax)
    fig = heatmap_figure(depth, SYSTEMS, S.CMAP_DIV, norm, num,
                         "well depth: min of mean PMF (kcal/mol)",
                         "Well depth of every PMF in every system (mean of halves; blue = attractive)",
                         text_dark=lambda v: v < -0.62 * vmax)
    cap = ("Each cell is the minimum of the mean-of-halves PMF (kcal/mol, zero at 1.8 nm) on a diverging scale "
           "centred at 0, so darker blue = deeper attractive well. Columns: the 36 residue pairs (residue on "
           "peptide 1 – residue on peptide 2), the 10 three-residue window pairs and the LIVA–LIVA fragment; bold "
           "labels involve residue 6 (X). Values and the half-to-half error at each minimum are in well_depths_all.csv.")
    return S.save(fig, f"{OUTDIR}/well_depth_heatmap.png", "Well depth heatmap (all systems, all PMFs)",
                  cap, GROUP, order=order, dpi=170)


def fig_well_position(df: pd.DataFrame, order: int):
    depth, rmin, _ = well_matrices(df)
    blank = (depth > NO_WELL) | (rmin >= TAIL_R)
    vals = rmin[~blank]
    norm = Normalize(float(np.floor(vals.min() * 20) / 20), float(np.ceil(vals.max() * 20) / 20))
    fig = heatmap_figure(rmin, SYSTEMS, S.CMAP_SEQ, norm, lambda v: f"{v:.2f}",
                         "distance of the minimum (nm)",
                         "Position of the PMF minimum (mean of halves)",
                         text_dark=lambda v: norm(v) > 0.55, blank=blank)
    cap = ("Each cell is the distance (nm) of the global minimum of the mean-of-halves PMF, i.e. the most "
           "favourable contact distance; darker = longer, '–' = no defined well (minimum shallower than "
           f"−{abs(NO_WELL):.1f} kcal/mol, or at r ≥ {TAIL_R:g} nm in the flat tail next to the 1.8 nm zero). "
           "Where a PMF has two wells of similar depth the position can jump between "
           "them across systems (e.g. L1–L1 at ~0.54 vs ~1.05 nm). Same columns as the well-depth heatmap; data in "
           "well_depths_all.csv.")
    return S.save(fig, f"{OUTDIR}/well_position_heatmap.png", "Well position heatmap (all systems, all PMFs)",
                  cap, GROUP, order=order, dpi=170)


def fig_delta_ref(df: pd.DataFrame, order: int):
    depth, _, err = well_matrices(df)
    ref = [k for k, s in enumerate(SYSTEMS) if s.key == REF_KEY][0]
    rows = [s for s in SYSTEMS if s.key != REF_KEY]
    idx = [k for k, s in enumerate(SYSTEMS) if s.key != REF_KEY]
    delta = depth[idx] - depth[ref]
    comb = np.sqrt(err[idx] ** 2 + err[ref] ** 2)
    muted = np.abs(delta) <= comb
    vmax = float(np.ceil(np.nanmax(np.abs(delta)) * 10) / 10)
    norm = Normalize(-vmax, vmax)
    fig = heatmap_figure(delta, rows, S.CMAP_DIV, norm, lambda v: num(v, signed=True),
                         "Δ well depth vs LIVAGG (kcal/mol)",
                         "Well depth relative to LIVAGG (X = Gly): blue = deeper than Gly, red = shallower",
                         text_dark=lambda v: abs(v) > 0.62 * vmax, muted=muted,
                         note="faded cell:\n|Δ| within the\ncombined\nhalves error")
    cap = ("Well depth (minimum of the mean-of-halves PMF) of each system minus that of LIVAGG, whose residue 6 "
           "is glycine (no side chain), isolating the effect of the X side chain and its charge: blue = deeper / "
           "more attractive than with Gly, red = shallower. Faded cells with gray values are within the combined "
           "half-to-half error at the two minima (|Δ| ≤ √(err² + err_G²), err = |PMF₁−PMF₂|/2, roughly one "
           "standard error) and should not be over-read. Same columns as the well-depth heatmap; data in "
           "well_depths_all.csv.")
    return S.save(fig, f"{OUTDIR}/well_depth_vs_livagg.png", "Well depth relative to LIVAGG (X = Gly)",
                  cap, GROUP, order=order, dpi=170)


# ------------------------------------------------------------------ main
def main() -> None:
    S.apply()
    store = Store()
    df = well_table(store)
    out_csv = C.assert_inside_overall(C.FIGURES / OUTDIR / "well_depths_all.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("wrote", out_csv.relative_to(C.OVERALL), len(df), "rows")
    for f in (lambda: fig_pairs(store, "mean", 0), lambda: fig_pairs(store, "full", 1),
              lambda: fig_windows(store, 2), lambda: fig_fragment(store, 3),
              lambda: fig_x_focus(store, 4), lambda: fig_livage(store, 5),
              lambda: fig_well_depth(df, 6), lambda: fig_well_position(df, 7),
              lambda: fig_delta_ref(df, 8)):
        p = f()
        print("wrote", p.relative_to(C.OVERALL), f"{p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
