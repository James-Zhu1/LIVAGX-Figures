"""Per-system overview: one row per LIVAGX system, seven columns.

    1  6x6 residue-pair PMFs        mean of halves, band = |PMF1-PMF2|/2
    2  three-residue window PMFs    4x4 upper triangle (LIV IVA VAG AGX), same
    3  LIVA-LIVA fragment PMF       same
    4  H-bond occupancy heatmap     4x4 LIVA positions, mean of halves (± half diff)
    5  H-bond network diagram       directed donor -> acceptor, width = mean occupancy
    6  pep1-pep2 angle              mean of the two half densities, band = |h1-h2|/2
    7  end-to-end distance          same

Output: overall/figures/overview/system_rows.png (group 'overview').

Run from overall/:
    /hopper/home/jzii2024/miniconda3/envs/pmf/bin/python scripts/fig_system_rows.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch  # noqa: E402

import style  # noqa: E402
from common import (SYSTEMS, WINDOW_PAIRS, load_fragment, load_pair, load_window,  # noqa: E402
                    pair_ids)
from fig_halved_compare import ANGLE_CAVEAT, atom_res, load_all, res_to_pos  # noqa: E402

XLIM = (0.3, 1.8)
YLIM = (-1.75, 2.5)
BAND_ALPHA = 0.3
WIN_NAMES = ["LIV", "IVA", "VAG", "AGX"]
E2E_BINS = np.arange(3.0, 19.0 + 1e-9, 0.2)
ANG_BINS = np.arange(0.0, 180.0 + 1e-9, 3.0)
MIN_EDGE = 0.005  # H-bond edges below this mean occupancy are not drawn
# H-bond direction colours: system colour = peptide 1 donates, gray = peptide 2 donates
HB_GRAY = "#6b6a66"


def donor_cmap(s, donor: int) -> LinearSegmentedColormap:
    """Light -> dark ramp in the system colour (pep 1 donates) or gray (pep 2 donates)."""
    if donor == 2:
        return LinearSegmentedColormap.from_list("hb_gray", ["#f4f3f0", "#b9b8b1", "#6b6a66", "#2f2e2b"])
    c = np.array(to_rgb(s.color))
    return LinearSegmentedColormap.from_list(f"hb_{s.key}", [(0.98, 0.98, 0.975), tuple(0.5 * c + 0.5), tuple(c),
                                                             tuple(0.55 * c)])


def donor_edge(s, donor: int) -> str:
    return s.color if donor == 1 else HB_GRAY

ROW_H = 3.3
COL_W = [3.3, 3.3, 3.3, 4.6, 3.0, 3.0, 3.0]
LABEL_W = 2.5


def mini_panel(ax, curve, color, labels: bool) -> None:
    ok = np.isfinite(curve.pmf)
    r, m = curve.r[ok], curve.pmf[ok]
    e = np.nan_to_num(curve.err[ok]) if curve.err is not None else np.zeros_like(m)
    ax.fill_between(r, m - e, m + e, color=color, alpha=BAND_ALPHA, lw=0)
    ax.plot(r, m, color=color, lw=0.8)
    ax.axhline(0, color=style.AXIS, lw=0.4, zorder=0)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([0.4, 1.0, 1.6])
    ax.set_yticks([-1, 0, 1, 2])
    ax.grid(lw=0.3)
    ax.tick_params(length=1.2, pad=1, labelsize=5)
    if not labels:
        ax.tick_params(labelbottom=False, labelleft=False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)


def pair_block(fig, cell, s, curves) -> None:
    g = cell.subgridspec(6, 6, hspace=0.06, wspace=0.06)
    for i, j in pair_ids():
        ax = fig.add_subplot(g[i - 1, j - 1])
        mini_panel(ax, curves[(i, j)], s.color, labels=(i == 6 and j == 1))
        if i == 1:
            ax.set_title(f"{s.sequence[j - 1]}{j}", fontsize=7, pad=1.5)
        if j == 1:
            ax.set_ylabel(f"{s.sequence[i - 1]}{i}", fontsize=7, rotation=0, ha="right",
                          va="center", labelpad=9 if i == 6 else 2)


def window_block(fig, cell, s, curves) -> None:
    g = cell.subgridspec(4, 4, hspace=0.08, wspace=0.08)
    names = [n.replace("X", s.x) for n in WIN_NAMES]
    for a, b in WINDOW_PAIRS:
        ax = fig.add_subplot(g[a - 1, b - 1])
        mini_panel(ax, curves[(a, b)], s.color, labels=(a == 4 and b == 4))
        if a == 1:
            ax.set_title(names[b - 1], fontsize=7, pad=1.5)
        if a == b:
            ax.set_ylabel(names[a - 1], fontsize=7, rotation=0, ha="right", va="center",
                          labelpad=9 if a == 4 else 2)


def fragment_panel(ax, s, c, bottom: bool) -> None:
    ok = np.isfinite(c.pmf)
    ax.fill_between(c.r[ok], (c.pmf - c.err)[ok], (c.pmf + c.err)[ok], color=s.color,
                    alpha=BAND_ALPHA, lw=0, label="±|PMF₁−PMF₂|/2")
    ax.plot(c.r[ok], c.pmf[ok], color=s.color, lw=style.LINE_W, label="mean of halves")
    k = int(np.nanargmin(c.pmf))
    ax.plot([c.r[k]], [c.pmf[k]], "o", ms=4, color=s.color, mec=style.SURFACE, mew=0.8)
    ax.annotate(f"{c.pmf[k]:.2f} ± {c.err[k]:.2f}\nat {c.r[k]:.2f} nm".replace("-", "−"),
                (c.r[k], c.pmf[k]), xytext=(0.97, 0.95), textcoords="axes fraction",
                ha="right", va="top", fontsize=7, color=style.INK_2)
    ax.axhline(0, color=style.AXIS, lw=0.6, zorder=0)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_ylabel(style.PMF_LABEL, fontsize=8)
    if bottom:
        ax.set_xlabel("COM distance (nm)", fontsize=8)


def hbond_halves(per) -> tuple[np.ndarray, np.ndarray, dict]:
    """Directed 4x4 occupancy matrices, as in the original hbond heatmaps.

    Returns mean and |h1-h2|/2 arrays of shape (2, 4, 4): [0] = peptide 1 donates
    (rows = donor position on pep 1, cols = acceptor position on pep 2), [1] =
    peptide 2 donates (rows = donor on pep 2, cols = acceptor on pep 1); plus the
    mean directed edge occupancies for the network diagram.
    """
    mats = np.zeros((2, 2, 4, 4))  # half, direction, donor pos, acceptor pos
    edges: dict[tuple, list[float]] = {}
    for k, h in enumerate(("first", "second")):
        for _, r in per[h]["hb"].iterrows():
            a, d = res_to_pos(atom_res(r.acceptor)), res_to_pos(atom_res(r.donor))
            if a is None or d is None or a[0] == d[0]:
                continue
            edges.setdefault((d, a), [0.0, 0.0])[k] += r.frac  # donor -> acceptor
            if d[1] <= 4 and a[1] <= 4:
                mats[k, d[0] - 1, d[1] - 1, a[1] - 1] += r.frac
    mean_edges = {k: 0.5 * (v[0] + v[1]) for k, v in edges.items()}
    return 0.5 * (mats[0] + mats[1]), 0.5 * np.abs(mats[0] - mats[1]), mean_edges


def hbond_heat(ax, s, mean, diff, norm, donor: int, bottom: bool) -> None:
    acceptor = 3 - donor
    cmap = donor_cmap(s, donor)
    ax.imshow(mean, cmap=cmap, norm=norm, aspect="equal")
    labs = [f"{s.sequence[p]}{p + 1}" for p in range(4)]
    ax.set_xticks(range(4), labs, fontsize=7)
    ax.set_yticks(range(4), labs, fontsize=7)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0, pad=2)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(4):
        for j in range(4):
            v = mean[i, j]
            r_, g_, b_ = cmap(norm(v))[:3]
            ink = "white" if 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_ < 0.45 else style.INK
            ax.text(j, i, f"{v:.2f}\n±{diff[i, j]:.2f}", ha="center", va="center",
                    fontsize=5.6, color=ink, linespacing=1.0)
    ax.set_title(f"pep {donor} → pep {acceptor}", fontsize=8, pad=14)
    ax.set_ylabel(f"donor (pep {donor})", fontsize=7, color=style.INK_2, labelpad=1)
    ax.set_xlabel(f"acceptor (pep {acceptor})", fontsize=7, color=style.INK_2, labelpad=2)


def hbond_network(ax, s, edges, wmax) -> None:
    """Circular layout from the original combined_first_second_half.py: the eight
    LIVA residues (pep 1 L I V A, then pep 2 L I V A) evenly spaced on a circle."""
    order = [(1, p) for p in range(1, 5)] + [(2, p) for p in range(1, 5)]
    theta = np.linspace(0, 2 * np.pi, len(order), endpoint=False)
    xy = {node: (np.cos(t), np.sin(t)) for node, t in zip(order, theta)}
    for (pep, p), (x, y) in xy.items():
        face = s.color if pep == 1 else style.SURFACE
        ax.scatter([x], [y], s=330, c=face, edgecolors=s.color, linewidths=1.6, zorder=4)
        ax.text(x, y, f"{s.sequence[p - 1]}{p}", ha="center", va="center", fontsize=7, zorder=5,
                color="white" if pep == 1 else style.INK, weight="bold")
    for ((dp, dpos), (ap, apos)), w in sorted(edges.items(), key=lambda kv: kv[1]):
        if w < MIN_EDGE or dpos > 4 or apos > 4:
            continue
        arr = FancyArrowPatch(xy[(dp, dpos)], xy[(ap, apos)], connectionstyle="arc3,rad=0.2",
                              arrowstyle="-|>", mutation_scale=6 + 6 * w / wmax,
                              lw=0.4 + 5.5 * w / wmax, color=donor_edge(s, dp), alpha=0.35 + 0.6 * w / wmax,
                              shrinkA=10, shrinkB=10, zorder=3)
        ax.add_patch(arr)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def density_panel(ax, s, d1, d2, bins, xlim, xlabel, bottom: bool, ymax) -> None:
    c = 0.5 * (bins[1:] + bins[:-1])
    h1 = np.histogram(d1, bins=bins, density=True)[0]
    h2 = np.histogram(d2, bins=bins, density=True)[0]
    m, e = 0.5 * (h1 + h2), 0.5 * np.abs(h1 - h2)
    ax.fill_between(c, m - e, m + e, color=s.color, alpha=BAND_ALPHA, lw=0)
    ax.plot(c, m, color=s.color, lw=style.LINE_W)
    mu = 0.5 * (np.mean(d1) + np.mean(d2))
    ax.axvline(mu, color=style.INK_2, lw=0.6)
    ax.text(0.97, 0.95, f"mean {mu:.1f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=style.INK_2)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, ymax)
    ax.tick_params(labelsize=7)
    ax.set_ylabel("density", fontsize=8)
    if bottom:
        ax.set_xlabel(xlabel, fontsize=8)


def build():
    data = load_all()
    hb = {s.key: hbond_halves(data[s.key]) for s in SYSTEMS}
    hb_vmax = max(v[0].max() for v in hb.values())
    norm = Normalize(0, hb_vmax)
    wmax = max(max(v[2].values()) for v in hb.values())

    def dens_max(key, bins):
        return 1.1 * max(np.histogram(data[s.key][h][key], bins=bins, density=True)[0].max()
                         for s in SYSTEMS for h in ("first", "second"))
    e2e_ymax, ang_ymax = dens_max("e2e", E2E_BINS), dens_max("angle", ANG_BINS)

    fig_w = LABEL_W + sum(COL_W) + 0.5 * len(COL_W)
    fig_h = ROW_H * len(SYSTEMS) + 0.4 * len(SYSTEMS) + 1.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(len(SYSTEMS), len(COL_W), width_ratios=COL_W,
                          left=LABEL_W / fig_w, right=1 - 0.15 / fig_w,
                          top=1 - 1.25 / fig_h, bottom=0.45 / fig_h, wspace=0.16, hspace=0.14)

    heads = ["Residue-pair PMFs (6×6)", "Three-residue window PMFs", "LIVA–LIVA fragment PMF",
             "H-bond occupancy (LIVA), by direction", "H-bond network", "pep1–pep2 angle", "End-to-end distance (pep 1)"]
    subs = ["each mini panel: x 0.3–1.8 nm, y −1.75…2.5 kcal/mol", "same axes; rows/cols = window on pep 1 / pep 2",
            "mean of halves ± |PMF₁−PMF₂|/2", "rows = donor, cols = acceptor; per contact frame, mean ± |h₁−h₂|/2",
            "filled = pep 1, open = pep 2; arrow = donor N–H → acceptor O", "mean of half densities ± |h₁−h₂|/2", "mean of half densities ± |h₁−h₂|/2"]

    for r, s in enumerate(SYSTEMS):
        bottom = r == len(SYSTEMS) - 1
        pair_block(fig, gs[r, 0], s, {ij: load_pair(s, *ij, kind="mean") for ij in pair_ids()})
        window_block(fig, gs[r, 1], s, {ab: load_window(s, *ab) for ab in WINDOW_PAIRS})
        fragment_panel(fig.add_subplot(gs[r, 2]), s, load_fragment(s), bottom)
        mean, diff, edges = hb[s.key]
        gh = gs[r, 3].subgridspec(1, 2, wspace=0.35)
        for k in range(2):
            hbond_heat(fig.add_subplot(gh[0, k]), s, mean[k], diff[k], norm, donor=k + 1, bottom=bottom)
        hbond_network(fig.add_subplot(gs[r, 4]), s, edges, wmax)
        d = data[s.key]
        density_panel(fig.add_subplot(gs[r, 5]), s, d["first"]["angle"], d["second"]["angle"],
                      ANG_BINS, (0, 180), "angle (deg)", bottom, ang_ymax)
        density_panel(fig.add_subplot(gs[r, 6]), s, d["first"]["e2e"], d["second"]["e2e"],
                      E2E_BINS, (3, 19), "end-to-end distance (Å)", bottom, e2e_ymax)

        # row label with system colour bar
        box = gs[r, 0].get_position(fig)
        yc = 0.5 * (box.y0 + box.y1)
        fig.add_artist(Line2D([0.12 / fig_w] * 2, [box.y0 + 0.01, box.y1 - 0.01], color=s.color, lw=5,
                              transform=fig.transFigure))
        fig.text(0.28 / fig_w, yc + 0.012, s.label, fontsize=12, weight="bold", va="center")
        fig.text(0.28 / fig_w, yc - 0.006, s.x_resname, fontsize=8.5, color=style.INK_2, va="center")

    for c, (h, sub) in enumerate(zip(heads, subs)):
        box = gs[0, c].get_position(fig)
        xc = 0.5 * (box.x0 + box.x1)
        fig.text(xc, 1 - 0.55 / fig_h, h, ha="center", fontsize=12, weight="bold")
        fig.text(xc, 1 - 0.8 / fig_h, sub, ha="center", fontsize=8, color=style.INK_2)

    # legends for the H-bond columns (once, under the column headers)
    leg = [Line2D([], [], color=style.INK, lw=2.5, label="pep 1 donates (system colour)"),
           Line2D([], [], color=HB_GRAY, lw=2.5, label="pep 2 donates (gray)")]
    box = gs[0, 4].get_position(fig)
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5 * (box.x0 + box.x1), 1 - 0.88 / fig_h),
               ncol=2, fontsize=7, handlelength=1.4, columnspacing=1.0)
    box = gs[len(SYSTEMS) - 1, 3].get_position(fig)
    bw = 0.42 * (box.x1 - box.x0)
    for k, (donor, x0) in enumerate(((1, box.x0 + 0.04 * (box.x1 - box.x0)), (2, box.x0 + 0.54 * (box.x1 - box.x0)))):
        cax = fig.add_axes([x0, 0.12 / fig_h, bw, 0.1 / fig_h])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=donor_cmap(SYSTEMS[-1], donor)), cax=cax,
                          orientation="horizontal")
        cb.set_label(f"pep {donor} → pep {3 - donor} occupancy" + (" (each row in its system colour)" if donor == 1 else ""), fontsize=7)
        cb.ax.tick_params(labelsize=6)
    return fig


CAPTION = (
    "One row per system, seven analyses. Columns 1–3: residue-pair (6×6, row = position on peptide 1, "
    "column = position on peptide 2), three-residue window (upper triangle) and LIVA–LIVA PMFs, all the "
    "mean of the two trajectory halves with a ±|PMF₁−PMF₂|/2 band; every PMF panel shares x 0.3–1.8 nm and "
    "y −1.75 to 2.5 kcal/mol (−kT ln P(r), zeroed at 1.8 nm, no r² Jacobian). Column 4: inter-peptide "
    "backbone H-bond occupancy over the LIVA positions (the cpptraj mask), split by direction as in the original "
    "hbond heatmaps (left: peptide 1 N–H donates to peptide 2 O; right: peptide 2 donates to peptide 1; rows = "
    "donor, columns = acceptor; system colour = peptide 1 donates, gray = peptide 2 donates, both on the same 0–max range), per LIVA–LIVA contact frame, mean ± half difference. Column 5: the same "
    "H-bonds as directed arrows (donor N–H → acceptor O; system colour = peptide 1 donates, gray = peptide 2) on the circular layout of the original analysis script (filled nodes = peptide 1, open = peptide 2); width and opacity ∝ mean occupancy on one scale "
    f"for all rows, bonds below {MIN_EDGE} not drawn. Columns 6–7: pep1–pep2 vector angle (contact frames) "
    "and peptide-1 end-to-end Cα distance (all frames), mean of the two half densities with ±|h₁−h₂|/2 band, "
    "vertical line = mean; shared axes down each column. " + ANGLE_CAVEAT)


def main() -> None:
    style.apply()
    fig = build()
    p = style.save(fig, "overview/system_rows.png", "Per-system overview: PMFs, H-bonds, angle, end-to-end",
                   CAPTION, "overview", [s.key for s in SYSTEMS], order=2, dpi=110)
    print("wrote", p, f"{p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
