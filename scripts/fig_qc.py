"""Convergence / quality-control figures: first half vs second half of every PMF.

For each of the 8 systems and each of the 47 PMFs (36 single-residue pairs,
10 three-residue window pairs, the LIVA-LIVA fragment) the two trajectory
halves (frames after the 5 % trim, see pmf_from_cpptraj.py) are put on a
common grid with common.combine_halves - the same algorithm as the source
script - and compared.

Outputs (group 'qc', all under overall/figures/qc/):
  halves_rmsd_heatmap.png    RMS(half 1 - half 2) per system x PMF       (order 0)
  depth_half1_vs_half2.png   well depth half 1 vs half 2, one panel/system (order 1)
  full_vs_mean.png           pair well depth: full trajectory vs mean of halves (order 2)
  convergence_table.csv      the numbers behind all three figures

Run from overall/:
  /hopper/home/jzii2024/miniconda3/envs/pmf/bin/python scripts/fig_qc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import common as C  # noqa: E402
import style as S  # noqa: E402

GROUP = "qc"
WALL_CUTOFF = 2.0  # kcal/mol: RMS only where the mean PMF is below this
FLAG_THRESHOLD = 0.3  # kcal/mol: |depth_half1 - depth_half2| above this is flagged
EDGE_PMF = 0.5  # kcal/mol: inner-side bins with mean PMF >= this = repulsive edge
EDGE_DOMINATED = 0.5  # mark heatmap cells where the edge gives > half of sum (h1-h2)^2
FAMILY_MARKER = {"pair": "o", "window": "s", "fragment": "D"}
FAMILY_TITLE = {"pair": "single-residue pairs (pep 1 – pep 2)",
                "window": "three-residue windows",
                "fragment": "fragment"}


# ------------------------------------------------------------------ numbers
def _finite(half):
    r, p = half
    ok = np.isfinite(r) & np.isfinite(p)
    return r[ok], p[ok]


def halves_on_grid(half1, half2, bin_width):
    """Common grid, both halves interpolated on it, and their mean.

    The grid and mean come from common.combine_halves (source algorithm); the
    halves are re-interpolated with the same finite-bin mask it uses, so that
    mean == (h1 + h2) / 2 exactly."""
    grid, mean, _ = C.combine_halves(half1, half2, bin_width)
    if grid.size == 0:
        return grid, grid, grid, grid
    a = np.interp(grid, *_finite(half1))
    b = np.interp(grid, *_finite(half2))
    return grid, a, b, mean


def qc_row(system, pmf_id, family, curve, bin_width, full=None):
    grid, a, b, mean = halves_on_grid(curve.half1, curve.half2, bin_width)
    below = mean < WALL_CUTOFF
    d2 = (a - b) ** 2
    rms = float(np.sqrt(np.mean(d2[below]))) if below.any() else np.nan
    # Review fix: the 2 kcal/mol cutoff removes the hard-core wall but not its
    # sparsely sampled foot. "Edge" = bins inside the mean-PMF minimum (shorter r)
    # whose mean PMF is >= EDGE_PMF; edge_share = fraction of the summed squared
    # half difference (within the cutoff) that comes from those bins.
    if below.any():
        inner = np.arange(mean.size) < int(np.argmin(mean))
        edge = below & inner & (mean >= EDGE_PMF)
        tot = float(d2[below].sum())
        edge_share = float(d2[edge].sum() / tot) if tot > 0 else 0.0
        rest = below & ~edge
        rms_excl = float(np.sqrt(np.mean(d2[rest]))) if rest.any() else np.nan
    else:
        edge_share, rms_excl = np.nan, np.nan
    # per-half depth: minimum of the raw half PMF (r < 1.8 nm, zero at 1.8 nm)
    d1 = float(np.min(_finite(curve.half1)[1]))
    d2 = float(np.min(_finite(curve.half2)[1]))
    dm = float(np.min(mean)) if mean.size else np.nan
    df_ = C.well_depth(full)[0] if full is not None else np.nan
    return dict(system=system.key, pmf_id=pmf_id, family=family,
                rms_halves_kcal=rms, depth_half1=d1, depth_half2=d2,
                depth_mean=dm, depth_full=df_,
                flag=bool(abs(d1 - d2) > FLAG_THRESHOLD),
                rms_excl_edge_kcal=rms_excl, edge_share=edge_share,
                n_grid_below_cutoff=int(below.sum()))


def pmf_columns():
    """The 47 PMFs in display order: (pmf_id generic label, family, loader args)."""
    cols = [(C.pair_label(i, j).replace("–", "-"), "pair", (i, j)) for i, j in C.pair_ids()]
    cols += [(C.window_label(a, b).replace("–", "-"), "window", (a, b)) for a, b in C.WINDOW_PAIRS]
    cols += [("LIVA-LIVA", "fragment", ())]
    return cols


def build_table() -> pd.DataFrame:
    rows = []
    for s in C.SYSTEMS:
        for pmf_id, family, args in pmf_columns():
            if family == "pair":
                curve = C.load_pair(s, *args, kind="mean")
                full = C.load_pair(s, *args, kind="full")
                rows.append(qc_row(s, pmf_id, family, curve, C.PAIR_BIN, full))
            elif family == "window":
                rows.append(qc_row(s, pmf_id, family, C.load_window(s, *args), C.WINDOW_BIN))
            else:
                rows.append(qc_row(s, pmf_id, family, C.load_fragment(s), C.WINDOW_BIN))
    return pd.DataFrame(rows)


def pretty(pmf_id: str) -> str:
    return pmf_id.replace("-", "–")


def _signed(v: float, nd: int) -> str:
    """'+0.25' / '−0.24' with a true minus sign, matching the axis tick labels."""
    return f"{v:+.{nd}f}".replace("-", "−")


# ------------------------------------------------------------------ (a) heatmap
def _ink_for(rgba) -> str:
    r, g, b = to_rgb(rgba)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.55 else S.INK


def fig_heatmap(tab: pd.DataFrame) -> Path:
    cols = pmf_columns()
    ids = [c[0] for c in cols]
    fams = [c[1] for c in cols]
    mat = (tab.pivot(index="system", columns="pmf_id", values="rms_halves_kcal")
           .loc[[s.key for s in C.SYSTEMS], ids].to_numpy())
    share = (tab.pivot(index="system", columns="pmf_id", values="edge_share")
             .loc[[s.key for s in C.SYSTEMS], ids].to_numpy())
    edge_dom = np.nan_to_num(share, nan=0.0) > EDGE_DOMINATED
    nrow, ncol = mat.shape

    # x layout: unit cells, a gap between families
    gap = 0.7
    x0 = np.zeros(ncol)
    x = 0.0
    for k in range(ncol):
        if k > 0 and fams[k] != fams[k - 1]:
            x += gap
        x0[k] = x
        x += 1.0
    xc = x0 + 0.5
    xmax = x0[-1] + 1.0

    vmax = float(np.ceil(np.nanmax(mat) / 0.05) * 0.05)
    norm = Normalize(0.0, vmax)
    cmap = S.CMAP_SEQ

    cell_w, cell_h = 0.36, 0.40
    fig_w = xmax * cell_w + 2.6
    fig_h = nrow * cell_h + 2.55
    fig = plt.figure(figsize=(fig_w, fig_h))
    left, right = 1.75 / fig_w, 1 - 0.85 / fig_w
    bottom, top = 1.05 / fig_h, 1 - 1.0 / fig_h
    ax = fig.add_axes([left, bottom, right - left, top - bottom])
    cax = fig.add_axes([right + 0.18 / fig_w, bottom, 0.14 / fig_w, top - bottom])

    # worst 5 per system -> bold text
    worst = np.zeros_like(mat, dtype=bool)
    for r in range(nrow):
        order = np.argsort(-np.nan_to_num(mat[r], nan=-1))
        worst[r, order[:5]] = True

    pad = 0.04
    for r in range(nrow):
        for k in range(ncol):
            v = mat[r, k]
            if not np.isfinite(v):
                continue
            col = cmap(norm(v))
            ax.add_patch(plt.Rectangle((x0[k] + pad, r + pad), 1 - 2 * pad, 1 - 2 * pad,
                                       facecolor=col, edgecolor="none"))
            ax.text(xc[k], r + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=6.3,
                    color=_ink_for(col), fontweight="bold" if worst[r, k] else "normal")
            if worst[r, k]:
                ax.add_patch(plt.Rectangle((x0[k] + pad, r + pad), 1 - 2 * pad, 1 - 2 * pad,
                                           facecolor="none", edgecolor=S.INK, linewidth=0.9))
            if edge_dom[r, k]:  # corner tick: RMS mostly from the repulsive edge
                xr, yt, t = x0[k] + 1 - pad, r + pad, 0.27
                ax.add_patch(plt.Polygon([(xr - t, yt), (xr, yt), (xr, yt + t)], closed=True,
                                         facecolor=_ink_for(col), edgecolor="none"))

    # thin separators between the 6-pair blocks (pep-1 residue changes)
    for k in range(6, 36, 6):
        ax.plot([x0[k] - 0.0, x0[k] - 0.0], [0, nrow], color=S.MUTED, lw=0.6,
                solid_capstyle="butt")

    ax.set_xlim(-0.05, xmax + 0.05)
    ax.set_ylim(nrow, 0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(xc)
    ax.set_xticklabels([pretty(i) for i in ids], rotation=90, fontsize=7, color=S.INK_2)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks(np.arange(nrow) + 0.5)
    ax.set_yticklabels([s.label for s in C.SYSTEMS], fontsize=8.5, color=S.INK)
    ax.tick_params(axis="y", length=0, pad=19)
    # system identity chips (fixed colours) next to the row labels
    for r, s in enumerate(C.SYSTEMS):
        ax.add_patch(plt.Rectangle((-0.58, r + 0.3), 0.4, 0.4, facecolor=s.color,
                                   edgecolor="none", clip_on=False))

    # family headers with brackets
    for fam in ("pair", "window", "fragment"):
        ks = [k for k in range(ncol) if fams[k] == fam]
        a, b = x0[ks[0]], x0[ks[-1]] + 1.0
        ax.plot([a + 0.05, b - 0.05], [-0.42, -0.42], color=S.INK_2, lw=0.8,
                clip_on=False, solid_capstyle="butt")
        label = FAMILY_TITLE[fam]
        if fam == "pair":
            label += "  ·  0.01 nm bins"
        elif fam == "window":
            label += "  ·  0.02 nm bins"
        else:
            label = "LIVA–\nLIVA"
        ax.text((a + b) / 2, -0.55, label, ha="center", va="bottom", fontsize=8,
                color=S.INK, clip_on=False)
    # pep-1 residue sub-headers over the pair blocks
    for blk, letter in enumerate("LIVAGX"):
        a = x0[blk * 6]
        ax.text(a + 3.0, -0.1, f"peptide-1 residue {letter}{blk + 1}", ha="center",
                va="bottom", fontsize=6.8, color=S.INK_2, clip_on=False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("RMS(half 1 − half 2)  (kcal/mol)", color=S.INK_2, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)

    fig.suptitle("Half-to-half agreement of every PMF  ·  RMS over the region where mean PMF "
                 f"< {WALL_CUTOFF:g} kcal/mol",
                 x=left, ha="left", y=1 - 0.12 / fig_h, va="top", fontsize=10, color=S.INK)

    # key for the cell marks (bottom-left, below the rotated tick labels)
    n_dom = int(edge_dom.sum())
    # positions in inches, converted to figure fraction (dpi_scale_trans artists are
    # not shifted by bbox_inches="tight", transFigure ones are)
    kx, ky, ks = left * fig_w, 0.16, 0.11

    def _f(x_in, y_in):
        return (x_in / fig_w, y_in / fig_h)

    fig.add_artist(plt.Rectangle(_f(kx, ky - 0.065), 0.16 / fig_w, 0.13 / fig_h,
                                 transform=fig.transFigure,
                                 facecolor="none", edgecolor=S.INK, linewidth=0.9))
    fig.text(*_f(kx + 0.24, ky), "bold, boxed = 5 worst PMFs of each system",
             ha="left", va="center", fontsize=7.5, color=S.INK_2)
    kx2 = kx + 3.0
    fig.add_artist(plt.Polygon([_f(kx2, ky + 0.065), _f(kx2 + ks, ky + 0.065),
                                _f(kx2 + ks, ky + 0.065 - ks)],
                               closed=True, transform=fig.transFigure,
                               facecolor=S.INK, edgecolor="none"))
    fig.text(*_f(kx2 + 0.2, ky),
             f"corner mark = more than half of the squared difference comes from the repulsive "
             f"edge inside the well (mean PMF {EDGE_PMF:g}–{WALL_CUTOFF:g} kcal/mol): "
             f"{n_dom} of {mat.size} cells",
             ha="left", va="center", fontsize=7.5, color=S.INK_2)

    top = tab.loc[tab.rms_halves_kcal.idxmax()]
    caption = (
        "Each cell is the root-mean-square difference between the first-half and second-half PMF "
        "(5%-trimmed halves, put on a common grid with the source combine_pmfs algorithm), "
        f"computed only where the mean-of-halves PMF is below {WALL_CUTOFF:g} kcal/mol to cut off "
        "the hard-core wall; darker blue = the two halves disagree more, and the five worst PMFs of "
        "each system are boxed in bold. The cutoff does not remove the sparsely sampled foot of the "
        f"wall: in the {n_dom} corner-marked cells most of the RMS comes from the repulsive side inside "
        f"the well (mean PMF {EDGE_PMF:g}–{WALL_CUTOFF:g} kcal/mol, usually a few bins at r < 0.45 nm), "
        f"including the largest value ({C.BY_KEY[top.system].label} {pretty(top.pmf_id)}: "
        f"{top.rms_halves_kcal:.2f} kcal/mol, {top.rms_excl_edge_kcal:.2f} without the edge), so "
        "unmarked cells are the better guide to disagreement in the well itself. "
        "Columns: 36 residue pairs (position on peptide 1 – position on peptide 2, X = residue 6), "
        "10 three-residue window pairs and the LIVA–LIVA fragment; data from the *_halves_value CSVs."
    )
    return S.save(fig, "qc/halves_rmsd_heatmap.png",
                  "Convergence: RMS difference between trajectory halves", caption,
                  GROUP, order=0, dpi=170)


# ------------------------------------------------------------------ (b) depth h1 vs h2
def fig_depth_halves(tab: pd.DataFrame) -> Path:
    lo = float(np.floor(min(tab.depth_half1.min(), tab.depth_half2.min()) / 0.1) * 0.1) - 0.05
    hi = float(max(tab.depth_half1.max(), tab.depth_half2.max(), 0.0)) + 0.1
    lim = (lo, hi)
    line = np.array(lim)

    fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.9), sharex=True, sharey=True)
    for ax, s in zip(axes.flat, C.SYSTEMS):
        sub = tab[tab.system == s.key]
        ax.fill_between(line, line - FLAG_THRESHOLD, line + FLAG_THRESHOLD,
                        color=S.GRID, alpha=0.7, lw=0, zorder=0)
        ax.plot(line, line, color=S.MUTED, lw=S.THIN_W, zorder=1)
        for fam, mk in FAMILY_MARKER.items():
            f = sub[sub.family == fam]
            size = {"pair": 26, "window": 30, "fragment": 46}[fam]
            ax.scatter(f.depth_half1, f.depth_half2, s=size, marker=mk, facecolor=s.color,
                       edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3)
        # largest disagreement in this system
        d = (sub.depth_half2 - sub.depth_half1)
        k = d.abs().idxmax()
        row = sub.loc[k]
        n_flag = int(sub.flag.sum())
        above = row.depth_half2 > row.depth_half1
        tx, ty = (0.04, 0.96) if above else (0.96, 0.04)
        ax.annotate(f"largest |Δ|: {pretty(row.pmf_id)}\n"
                    f"Δ = {_signed(row.depth_half2 - row.depth_half1, 2)} kcal/mol\n"
                    f"{n_flag} of {len(sub)} PMFs with |Δ| > {FLAG_THRESHOLD:g}",
                    xy=(row.depth_half1, row.depth_half2), xycoords="data",
                    xytext=(tx, ty), textcoords="axes fraction",
                    ha="left" if above else "right", va="top" if above else "bottom",
                    fontsize=7.2, color=S.INK, linespacing=1.3,
                    arrowprops=dict(arrowstyle="-", color=S.INK_2, lw=0.6,
                                    shrinkA=2, shrinkB=4))
        ax.scatter([row.depth_half1], [row.depth_half2], s=70, facecolor="none",
                   edgecolor=S.INK, linewidth=0.9, marker=FAMILY_MARKER[row.family], zorder=4)
        ax.set_title(s.label, color=S.INK, fontsize=9.5, loc="left")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    for ax in axes[1]:
        ax.set_xlabel("well depth, half 1 (kcal/mol)")
    for ax in axes[:, 0]:
        ax.set_ylabel("well depth, half 2 (kcal/mol)")

    handles = [Line2D([], [], ls="none", marker=FAMILY_MARKER[f], markersize=6,
                      markerfacecolor=S.INK_2, markeredgecolor="white",
                      label={"pair": "single-residue pair (36)", "window": "three-residue window (10)",
                             "fragment": "LIVA–LIVA fragment"}[f]) for f in FAMILY_MARKER]
    handles += [Line2D([], [], color=S.MUTED, lw=S.THIN_W, label="y = x"),
                Patch(facecolor=S.GRID, alpha=0.7, label=f"|half 2 − half 1| ≤ {FLAG_THRESHOLD:g} kcal/mol"),
                Line2D([], [], ls="none", marker="o", markersize=8, markerfacecolor="none",
                       markeredgecolor=S.INK, label="black outline = largest |Δ| in panel")]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.005),
               fontsize=8, handletextpad=0.4, columnspacing=1.6)
    fig.suptitle("Well depth (minimum PMF) from the first vs the second half of the trajectory  ·  "
                 "47 PMFs per system", x=0.06, ha="left", fontsize=10.5, color=S.INK)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97), h_pad=2.2, w_pad=1.2)

    caption = (
        "Each point is one of the 47 PMFs of a system (marker shape = PMF family); x = minimum of the "
        "first-half PMF, y = minimum of the second-half PMF (kcal/mol, zero at 1.8 nm, so more negative "
        "= deeper attractive well). Points on the y = x line agree between halves (necessary, not "
        f"sufficient, for convergence); the gray band is the ±{FLAG_THRESHOLD:g} kcal/mol flag threshold, "
        "and the PMF with the largest half-to-half disagreement is outlined and named in each panel. "
        "Depths are single-bin minima of the raw half PMFs (0.01 nm pair / 0.02 nm window and fragment bins, from the "
        "*_halves_value CSVs), so part of the scatter is bin noise rather than drift."
    )
    return S.save(fig, "qc/depth_half1_vs_half2.png",
                  "Convergence: well depth, first half vs second half", caption,
                  GROUP, order=1, dpi=160)


# ------------------------------------------------------------------ (c) full vs mean
def fig_full_vs_mean(tab: pd.DataFrame) -> Path:
    pairs = tab[tab.family == "pair"].copy()
    pairs["resid"] = pairs.depth_full - pairs.depth_mean
    lo = float(np.floor(min(pairs.depth_full.min(), pairs.depth_mean.min()) / 0.1) * 0.1) - 0.05
    hi = float(max(pairs.depth_full.max(), pairs.depth_mean.max(), 0.0)) + 0.08
    line = np.array([lo, hi])

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.4, 8.0), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3.2, 1.2], hspace=0.08))
    ax.plot(line, line, color=S.MUTED, lw=S.THIN_W, zorder=1, label="y = x")
    axr.axhline(0, color=S.MUTED, lw=S.THIN_W, zorder=1)
    for s in C.SYSTEMS:
        sub = pairs[pairs.system == s.key]
        kw = dict(s=24, facecolor=s.color, edgecolor="white", linewidth=0.4, alpha=0.85, zorder=3)
        ax.scatter(sub.depth_mean, sub.depth_full, label=s.label, **kw)
        axr.scatter(sub.depth_mean, sub.resid, **kw)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("well depth, full trajectory (kcal/mol)")
    ax.legend(loc="upper left", ncol=1, fontsize=7.5, handletextpad=0.3, borderaxespad=0.6)
    rmax = float(np.ceil(pairs.resid.abs().max() / 0.02) * 0.02)
    rmax = max(rmax, 0.06)
    axr.set_ylim(-rmax, rmax)
    axr.set_ylabel("full − mean\n(kcal/mol)")
    axr.set_xlabel("well depth, mean of the two halves (kcal/mol)")
    rms = float(np.sqrt(np.mean(pairs.resid ** 2)))
    worst = pairs.loc[pairs.resid.abs().idxmax()]
    ax.text(0.97, 0.04,
            f"{len(pairs)} pair PMFs (36 × {pairs.system.nunique()} systems)\n"
            f"mean(full − mean) = {_signed(pairs.resid.mean(), 3)} kcal/mol\n"
            f"RMS(full − mean) = {rms:.3f} kcal/mol\n"
            f"max |full − mean| = {abs(worst.resid):.3f} kcal/mol\n"
            f"({C.BY_KEY[worst.system].label}, {pretty(worst.pmf_id)})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=S.INK,
            linespacing=1.35)
    ax.set_title("Pair-PMF well depth: full trajectory vs mean of halves",
                 loc="left", fontsize=10, color=S.INK)
    fig.align_ylabels([ax, axr])

    caption = (
        "For the 36 single-residue pair PMFs of every system, the minimum of the whole-trajectory PMF "
        "(single_residue_pmf_full_value, 5%-trimmed) is plotted against the minimum of the mean-of-halves "
        "PMF (single_residue_pmf_mean_value); colours are the fixed system colours. The lower panel shows "
        f"the residual full − mean: it is small (RMS {rms:.3f}, max {abs(worst.resid):.3f} kcal/mol) but "
        f"slightly negative on average ({_signed(pairs.resid.mean(), 3)} kcal/mol, {100 * (pairs.resid < 0).mean():.0f}% "
        "of points below 0). That direction is expected: before re-zeroing, −kT·ln of a pooled histogram is "
        "never above the average of the two half PMFs (Jensen), and the minimum of an averaged curve is "
        "never deeper than the average of the two minima. The full PMF also includes the 5% of frames "
        "trimmed from the start of the second half."
    )
    return S.save(fig, "qc/full_vs_mean.png",
                  "Pair well depth: full trajectory vs mean of halves", caption,
                  GROUP, order=2, dpi=170)


# ------------------------------------------------------------------ main
def main() -> None:
    S.apply()
    tab = build_table()
    out_csv = C.assert_inside_overall(C.FIGURES / "qc" / "convergence_table.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["system", "pmf_id", "family", "rms_halves_kcal", "depth_half1", "depth_half2",
            "depth_mean", "depth_full", "flag",
            # review addition: how much of rms_halves_kcal comes from the repulsive edge
            "rms_excl_edge_kcal", "edge_share"]
    tab[cols].to_csv(out_csv, index=False, float_format="%.4f", na_rep="NaN")
    print("wrote", out_csv)
    for f in (fig_heatmap(tab), fig_depth_halves(tab), fig_full_vs_mean(tab)):
        print("wrote", f, f"{f.stat().st_size / 1e6:.2f} MB")
    print(f"flagged PMFs (|Δdepth| > {FLAG_THRESHOLD} kcal/mol): {int(tab.flag.sum())} of {len(tab)}")
    if tab.flag.any():
        print(tab[tab.flag][["system", "pmf_id", "depth_half1", "depth_half2"]].to_string(index=False))
    print("min points below cutoff:", tab.n_grid_below_cutoff.min())
    print(f"cells with edge_share > {EDGE_DOMINATED}: {int((tab.edge_share > EDGE_DOMINATED).sum())}; "
          f"max RMS excluding the edge {tab.rms_excl_edge_kcal.max():.3f} kcal/mol")


if __name__ == "__main__":
    main()
