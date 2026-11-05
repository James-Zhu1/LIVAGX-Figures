#!/usr/bin/env python3
"""2D PMFs of the pep1-pep2 angle against each peptide's end-to-end distance.

On the LIVA-LIVA contact frames (both halves pooled, see e2e_angle_pmf.py):

  type A    angle x end-to-end of peptide A (peptide 1, :2@CA-:7@CA)
  type B    angle x end-to-end of peptide B (peptide 2, :10@CA-:15@CA)
  type avg  (F_A + F_B) / 2

Outputs (group 'e2e_angle', all under overall/figures/e2e_angle_2d/):
  pmf2d_all_systems.png    8 systems x 3 types on one numeric range (0-4 kcal/mol)
  pmf2d_<key>.png          one row per system (shown on that system's page)
  pmf2d_halves_qc.png      |F_half1 - F_half2| / 2 for A and B, and F_A - F_B
  pmf2d_surfaces.csv       every bin of every surface (+ counts, + half surfaces)
  pmf2d_summary.csv        lowest bin, mean end-to-end by orientation, half / A-B differences

Run from overall/:
    /hopper/home/jzii2024/miniconda3/envs/pmf/bin/python scripts/fig_e2e_angle_2d.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import e2e_angle_pmf as P  # noqa: E402
import style  # noqa: E402
from common import FIGURES, SYSTEMS, assert_inside_overall  # noqa: E402
from fig_halved_compare import ANGLE_CAVEAT  # noqa: E402

GROUP = "e2e_angle"
OUT = FIGURES / "e2e_angle_2d"
WELL = 2.0  # kcal/mol: bins below this on the pooled surface are "well sampled" for the QC numbers
QC_ABS_MAX = 1.0  # kcal/mol, colour range of the half-difference maps
QC_AB_MAX = 1.0  # kcal/mol, colour range of F_A - F_B
QC_COLS = (("A", "half"), ("B", "half"), ("avg", "ab"))  # (surface whose bins are shown / counted, map kind)

CHARGE_NOTE = ("In LIVAGE (E⁻/E⁰) peptide A carries the charged Glu (GLU) and peptide B the neutral one (GLH); "
               "in every other system the two peptides are identical.")
METHOD = (
    "F = −kT ln P(θ, d) on the LIVA–LIVA contact frames (centre-of-mass distance of :2-5 and :10-13 ≤ 7 Å), "
    f"half 1 (frames 30 001–300 000) and half 2 (330 001–600 000) pooled; {P.ANG_BIN:g}° × {P.E2E_BIN:g} Å bins, "
    "T = 300 K, each surface shifted so its lowest bin is 0, no Jacobian (sin θ or d²). θ = angle between "
    "v1 = :2→:5 (peptide 1) and v2 = :9→:12 (peptide 2) from pep1pep2_angle_*_half.dat; d = Cα–Cα end-to-end "
    "distance from SECTION 3 of the halved cpptraj input (end_to_end_dist_pepA.dat = :2@CA–:7@CA, "
    "end_to_end_dist_pepB.dat = :10@CA–:15@CA, both on filtered_frames_first_half.nc then _second_half.nc, "
    "matched row by row to the angle files). Average = (F_A + F_B)/2 bin by bin, then re-zeroed; empty where "
    "either surface is empty. " + CHARGE_NOTE)
READ = (f"Contours at {', '.join(f'{v:g}' for v in P.LEVELS)} kcal/mol; ringed dot = lowest bin (the basin floor "
        "is flat, several bins lie within ~0.1 kcal/mol of it, so read the basin rather than the dot; bin centre "
        f"and gap in pmf2d_summary.csv); colour capped at {P.FMAX:g} kcal/mol (lightest step = ≥ {P.FMAX:g} or "
        "never visited). Same numeric range in every row, each in its own system colour; the hues differ in "
        "lightness, so compare systems by the contour lines, not by shade.")


def row_label(fig, ax_first, s, fig_w: float, x_bar: float = 0.10, x_text: float = 0.26) -> None:
    box = ax_first.get_position()
    yc = 0.5 * (box.y0 + box.y1)
    fig.add_artist(Line2D([x_bar / fig_w] * 2, [box.y0 + 0.004, box.y1 - 0.004], color=s.color, lw=5,
                          transform=fig.transFigure))
    fig.text(x_text / fig_w, yc + 0.006, s.label, fontsize=10, weight="bold", va="center")
    fig.text(x_text / fig_w, yc - 0.006, s.x_resname, fontsize=8, color=style.INK_2, va="center")


# ====================================================================== figures
def fig_all(data: dict) -> Path:
    n = len(SYSTEMS)
    label_w, pw, ph, cbw = 2.1, 3.0, 2.55, 0.9
    fig_w = label_w + 3 * pw + 0.25 * 3 + cbw
    fig_h = n * ph + 0.2 * n + 1.3
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n, 4, width_ratios=[pw, pw, pw, 0.12], left=label_w / fig_w,
                          right=1 - 0.55 / fig_w, top=1 - 0.9 / fig_h, bottom=0.55 / fig_h,
                          wspace=0.08, hspace=0.12)
    for r, s in enumerate(SYSTEMS):
        surf = data[s.key]["surf"]
        bottom = r == n - 1
        axes = []
        for k, t in enumerate(P.TYPES):
            ax = fig.add_subplot(gs[r, k])
            P.draw(ax, s, surf[t], bottom=bottom, left=k == 0, fs=7.5)
            axes.append(ax)
        cax = fig.add_subplot(gs[r, 3])
        P.colorbar(fig, cax, s, "PMF (kcal/mol)", orientation="vertical", fs=7)
        row_label(fig, axes[0], s, fig_w)
    for k, t in enumerate(P.TYPES):
        box = gs[0, k].get_position(fig)
        xc = 0.5 * (box.x0 + box.x1)
        fig.text(xc, 1 - 0.38 / fig_h, P.TYPE_TITLE[t], ha="center", fontsize=11, weight="bold")
        fig.text(xc, 1 - 0.6 / fig_h, f"angle × end-to-end, {P.TYPE_MASK[t]}", ha="center", fontsize=8,
                 color=style.INK_2)
    caption = ("Angle × end-to-end 2D PMFs on LIVA–LIVA contact frames, one row per system; columns = end-to-end "
               "distance of peptide A (peptide 1), of peptide B (peptide 2) and their average. " + METHOD + " "
               + READ + " " + ANGLE_CAVEAT)
    return style.save(fig, "e2e_angle_2d/pmf2d_all_systems.png", "Angle × end-to-end 2D PMFs, all systems",
                      caption, GROUP, [s.key for s in SYSTEMS], order=0, dpi=200)


def fig_system(s, d: dict, order: int) -> Path:
    fig = plt.figure(figsize=(10.4, 3.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.05], left=0.07, right=0.985, top=0.86, bottom=0.12,
                          wspace=0.08, hspace=0.42)
    for k, t in enumerate(P.TYPES):
        ax = fig.add_subplot(gs[0, k])
        P.draw(ax, s, d["surf"][t], bottom=True, left=k == 0, fs=8)
        ax.set_title(f"{P.TYPE_TITLE[t]}  ·  {P.TYPE_MASK[t]}", fontsize=9, pad=4)
    cax = fig.add_subplot(gs[1, 1])
    P.colorbar(fig, cax, s, f"PMF (kcal/mol), 0 = lowest bin; contours {', '.join(f'{v:g}' for v in P.LEVELS)}", fs=7.5)
    cf = d["cf"]
    fig.suptitle(f"{s.label} — angle × end-to-end 2D PMF on contact frames "
                 f"({cf.n1:,} + {cf.n2:,} frames, halves pooled)", fontsize=11, x=0.07, ha="left", y=0.985)
    caption = (f"{s.label}: angle × end-to-end 2D PMFs on its {cf.n:,} LIVA–LIVA contact frames "
               f"({cf.n1:,} in half 1, {cf.n2:,} in half 2), for peptide A, peptide B and their average. "
               + METHOD + " " + READ + " " + ANGLE_CAVEAT)
    return style.save(fig, f"e2e_angle_2d/pmf2d_{s.key}.png", f"{s.label}: angle × end-to-end 2D PMF",
                      caption, GROUP, [s.key], order=10 + order, dpi=200)


def qc_maps(d: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per QC column: (map, mask of bins counted for the mean/max numbers). Differences use the
    unshifted -kT ln p (each half over its own frames; A and B over the same frames), so the
    per-surface zero of the plotted PMFs does not enter. Shared by the figure and the table."""
    surf, raw, r1, r2 = d["surf"], d["raw"], d["raw1"], d["raw2"]
    out = []
    for t, kind in QC_COLS:
        shown = np.isfinite(surf[t]) & (surf[t] < P.FMAX)
        m = 0.5 * np.abs(r1[t] - r2[t]) if kind == "half" else raw["A"] - raw["B"]
        out.append((np.where(shown, m, np.nan), shown & (surf[t] < WELL)))
    return out


def fig_qc(data: dict) -> Path:
    n = len(SYSTEMS)
    label_w, pw, ph = 2.1, 3.0, 2.55
    fig_w = label_w + 3 * pw + 0.25 * 3 + 0.2
    fig_h = n * ph + 0.2 * n + 1.9
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n, 3, left=label_w / fig_w, right=1 - 0.2 / fig_w, top=1 - 0.9 / fig_h,
                          bottom=1.25 / fig_h, wspace=0.08, hspace=0.12)
    seq = style.CMAP_SEQ.copy()
    seq.set_over(style.SEQ_BLUE[-1])
    div = style.CMAP_DIV.copy()
    norms = (Normalize(0, QC_ABS_MAX), Normalize(0, QC_ABS_MAX), Normalize(-QC_AB_MAX, QC_AB_MAX))
    cmaps = (seq, seq, div)
    heads = [("|F₁ − F₂| / 2, pep A", "half 1 vs half 2, each −kT ln p over its own frames"),
             ("|F₁ − F₂| / 2, pep B", "same, surface B"),
             ("F_A − F_B", "−kT ln(count_A / count_B); blue = more probable for pep A")]
    for r, s in enumerate(SYSTEMS):
        d = data[s.key]
        axes = []
        for k, (m, well) in enumerate(qc_maps(d)):
            ax = fig.add_subplot(gs[r, k])
            # masked bins are transparent over a hatched background = "not shown" (≠ zero difference)
            ax.patch.set_facecolor(style.SURFACE)
            ax.patch.set_hatch("////")
            ax.patch.set_edgecolor(style.GRID)
            ax.pcolormesh(P.ANG_EDGES, P.E2E_EDGES, np.ma.masked_invalid(m).T, cmap=cmaps[k], norm=norms[k],
                          shading="flat", rasterized=True)
            ax.contour(P.ANG_C, P.E2E_C, np.ma.masked_invalid(d["surf"]["avg"]).T, levels=P.LEVELS,
                       colors=style.INK, linewidths=0.35, alpha=0.5)
            v = np.abs(m[well])
            ax.text(*P.NOTE_XY, f"mean {np.nanmean(v):.2f}, max {np.nanmax(v):.2f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color=style.INK_2,
                    bbox=dict(boxstyle="square,pad=0.15", fc=style.SURFACE, ec="none", alpha=0.85))
            P.style_axes(ax, bottom=r == n - 1, left=k == 0, fs=7.5)
            axes.append(ax)
        row_label(fig, axes[0], s, fig_w)
    for k, (h, sub) in enumerate(heads):
        box = gs[0, k].get_position(fig)
        xc = 0.5 * (box.x0 + box.x1)
        fig.text(xc, 1 - 0.38 / fig_h, h, ha="center", fontsize=11, weight="bold")
        fig.text(xc, 1 - 0.6 / fig_h, sub, ha="center", fontsize=8, color=style.INK_2)
    for k, (cmap, norm, lab, ext) in enumerate(
            ((seq, norms[0], "half-to-half difference (kcal/mol)", "max"),
             (div, norms[2], "F_A − F_B (kcal/mol)", "both"))):
        if k == 0:
            x0, x1 = gs[n - 1, 0].get_position(fig).x0, gs[n - 1, 1].get_position(fig).x1
        else:
            box = gs[n - 1, 2].get_position(fig)
            x0, x1 = box.x0, box.x1
        cax = fig.add_axes([x0 + 0.15 * (x1 - x0), 0.35 / fig_h, 0.7 * (x1 - x0), 0.12 / fig_h])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend=ext)
        cb.set_label(lab, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_linewidth(0.4)
    caption = ("Convergence and peptide A vs B for the angle × end-to-end 2D PMFs. Left and middle: |F₁ − F₂|/2 "
               "between the surfaces built from half 1 and from half 2 alone, each −kT ln p normalised over its own "
               "frames (no per-surface zero, so no constant offset), for peptide A and peptide B. Right: "
               "F_A − F_B = −kT ln(count_A / count_B) on the pooled frames; blue = that (θ, d) is more probable "
               f"with peptide A's end-to-end distance. {CHARGE_NOTE} Even for identical peptides A − B is not pure "
               "noise: θ is not symmetric in the two peptides (v1 = :2→:5 on peptide 1, v2 = :9→:12 on peptide 2, "
               "see caveat), so it mixes sampling noise with that asymmetry. Bins are coloured where the column's "
               f"pooled surface (A, B, average) is below {P.FMAX:g} kcal/mol; hatched = not shown. Panel numbers = "
               f"mean and max |value| over bins below {WELL:g} kcal/mol on the same surface (the halves_* and "
               "A_minus_B_* columns of pmf2d_summary.csv). Contours = average surface at "
               f"{', '.join(f'{v:g}' for v in P.LEVELS)} kcal/mol. Same bins and definitions as the 2D PMFs.")
    return style.save(fig, "e2e_angle_2d/pmf2d_halves_qc.png", "Angle × end-to-end 2D PMF: halves and A vs B",
                      caption, GROUP, [s.key for s in SYSTEMS], order=1, dpi=200)


# ====================================================================== tables
def tables(data: dict) -> None:
    long, summ = [], []
    ia, ie = np.meshgrid(np.arange(len(P.ANG_C)), np.arange(len(P.E2E_C)), indexing="ij")
    ia, ie = ia.ravel(), ie.ravel()
    for s in SYSTEMS:
        d = data[s.key]
        cf, surf, cnt, h1, h2 = d["cf"], d["surf"], d["counts"], d["h1"], d["h2"]
        long.append(pd.DataFrame({
            "system": s.key,
            "angle_lo_deg": P.ANG_EDGES[ia], "angle_hi_deg": P.ANG_EDGES[ia + 1],
            "e2e_lo_angstrom": P.E2E_EDGES[ie], "e2e_hi_angstrom": P.E2E_EDGES[ie + 1],
            "count_A": cnt["A"].ravel().astype(int), "count_B": cnt["B"].ravel().astype(int),
            "pmf_A": surf["A"].ravel(), "pmf_B": surf["B"].ravel(), "pmf_avg": surf["avg"].ravel(),
            "pmf_A_half1": h1["A"].ravel(), "pmf_A_half2": h2["A"].ravel(),
            "pmf_B_half1": h1["B"].ravel(), "pmf_B_half2": h2["B"].ravel(),
        }))
        anti = cf.angle >= 90.0
        e_all = {"A": cf.eA, "B": cf.eB, "avg": np.concatenate([cf.eA, cf.eB])}
        anti_all = {"A": anti, "B": anti, "avg": np.concatenate([anti, anti])}
        (mA, wA), (mB, wB), (mAB, wAB) = qc_maps(d)
        half = {"A": np.abs(mA[wA]), "B": np.abs(mB[wB])}
        ab = np.abs(mAB[wAB])
        for t in P.TYPES:
            th, dd = P.minimum(surf[t])
            e, a = e_all[t], anti_all[t]
            hd = half.get(t)
            summ.append(dict(
                system=s.key, label=s.label, type=t, e2e_mask=P.TYPE_MASK[t],
                n_frames=cf.n, n_half1=cf.n1, n_half2=cf.n2,
                min_angle_deg=round(th, 2), min_e2e_angstrom=round(dd, 2),
                min_gap_kcal=round(P.min_gap(surf[t]), 4),
                mean_e2e_angstrom=round(float(e.mean()), 3),
                mean_e2e_angstrom_angle_lt_90=round(float(e[~a].mean()), 3),
                mean_e2e_angstrom_angle_ge_90=round(float(e[a].mean()), 3),
                frac_angle_ge_90=round(float(anti.mean()), 4),
                halves_mean_abs_diff_div2_kcal=round(float(np.nanmean(hd)), 3) if hd is not None else np.nan,
                halves_max_abs_diff_div2_kcal=round(float(np.nanmax(hd)), 3) if hd is not None else np.nan,
                A_minus_B_mean_abs_kcal=round(float(np.nanmean(ab)), 3) if t == "avg" else np.nan,
                A_minus_B_max_abs_kcal=round(float(np.nanmax(ab)), 3) if t == "avg" else np.nan,
            ))
    f1 = assert_inside_overall(OUT / "pmf2d_surfaces.csv")
    pd.concat(long, ignore_index=True).to_csv(f1, index=False, float_format="%.4f")
    f2 = assert_inside_overall(OUT / "pmf2d_summary.csv")
    pd.DataFrame(summ).to_csv(f2, index=False)
    print("wrote", f1)
    print("wrote", f2)
    print(pd.DataFrame(summ)[["system", "type", "n_frames", "min_angle_deg", "min_e2e_angstrom", "min_gap_kcal",
                              "mean_e2e_angstrom", "mean_e2e_angstrom_angle_lt_90", "mean_e2e_angstrom_angle_ge_90",
                              "halves_mean_abs_diff_div2_kcal", "A_minus_B_mean_abs_kcal"]].to_string(index=False))


def load_all() -> dict:
    data = {}
    for s in SYSTEMS:
        cf = P.load(s)
        surf, counts, raw = P.surfaces(cf)
        (h1, raw1), (h2, raw2) = P.half_surfaces(cf)
        data[s.key] = dict(cf=cf, surf=surf, counts=counts, raw=raw, h1=h1, h2=h2, raw1=raw1, raw2=raw2)
    return data


def main() -> None:
    style.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all()
    print("wrote", fig_all(data))
    for i, s in enumerate(SYSTEMS):
        print("wrote", fig_system(s, data[s.key], i))
    print("wrote", fig_qc(data))
    tables(data)


if __name__ == "__main__":
    main()
