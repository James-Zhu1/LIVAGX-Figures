#!/usr/bin/env python3
"""Cross-system comparison of the halved (first vs second half) structural analyses.

Source: each system's analysis_continued/halved/ folder, produced by
hbond_multidihedral_angle.in (cpptraj) on com_frames_nowater.dcd:

  half 1 = frames  30001-300000, half 2 = frames 330001-600000 (270 000 frames each)

  pep1pep2_angle_{h}_half.dat   one row per LIVA-LIVA *contact* frame (cpptraj
                                `distance :2-5 :10-13` = centre-of-mass distance
                                of the two LIVA segments, filtered to < 7 A); angle between
                                v1 = :2 -> :5 and v2 = :9 -> :12 (NB: v2 starts at
                                ACE 9, probably an off-by-one for :10 -> :13)
  interhbond_{h}_half.txt       cpptraj hbond avgout, mask :2-5,10-13 nointramol,
                                contact frames only (Frac = frames / contact frames)
  multidihedral_{h}_half.dat    phi/psi of residues 2-7 and 10-15, contact frames
                                only (too large for the snapshot: read read-only
                                from the source folder)
  end_to_end_dist_{h}_half.dat  :2@CA - :7@CA of peptide 1, all frames

Outputs (group 'halved', all under overall/figures/halved/):
  contact_frames.png, hbond_maps.png, hbond_halves_diff.png, hbond_total.png,
  end_to_end.png, pep_angle.png, ramachandran_grid.png,
  secondary_structure_populations.png,
  halved_summary.csv, hbond_occupancy.csv, secondary_structure_populations.csv

Run from overall/:
    python scripts/fig_halved_compare.py            # everything
    python scripts/fig_halved_compare.py --no-dihedrals   # skip the 400 MB read
"""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory, offset_copy

import style
from common import (FIGURES, SYSTEMS, assert_inside_overall, halved_dir,
                    load_cpptraj_dat, load_hbonds)

GROUP = "halved"
OUT = FIGURES / "halved"
ALL_KEYS = [s.key for s in SYSTEMS]

HALVES = ("first", "second")
HALF_NAME = {"first": "half 1", "second": "half 2"}
HALF_FRAMES = {"first": "frames 30 001–300 000", "second": "frames 330 001–600 000"}
GENERIC = "LIVAGX"
LIVA_POS = [1, 2, 3, 4]  # the hbond mask :2-5,10-13 only contains L I V A
POS_LABELS = [f"{GENERIC[p - 1]}{p}" for p in range(1, 7)]

ANGLE_CAVEAT = ("Caveat: in the source cpptraj input the second vector is :9→:12 "
                "(ACE→Val of peptide 2) while the first is :2→:5 (Leu→Ala of peptide 1), "
                "probably an off-by-one for :10→:13, so the angle is between two slightly "
                "different segments.")

# simple published phi/psi region definitions (degrees), mutually exclusive here
SS_REGIONS = {
    "β": "φ ∈ [−180, −90), ψ ∈ [90, 180] ∪ [−180, −150)",
    "PPII": "φ ∈ [−90, −45), ψ ∈ [90, 180]",
    "αR": "φ ∈ [−160, −20), ψ ∈ [−120, 50)",
    "other": "everything else (incl. αL)",
}


# ====================================================================== helpers
def res_to_pos(resnum: int) -> tuple[int, int] | None:
    """prmtop residue number -> (peptide 1|2, position 1..6)."""
    if 2 <= resnum <= 7:
        return 1, resnum - 1
    if 10 <= resnum <= 15:
        return 2, resnum - 9
    return None


def atom_res(token: str) -> int:
    """'ILE_11@O' -> 11"""
    return int(token.split("@")[0].rsplit("_", 1)[1])


def hbond_matrix(hb: pd.DataFrame) -> tuple[np.ndarray, float, int]:
    """Symmetric 4x4 occupancy matrix over LIVA positions.

    Cell (i, j) = sum of Frac of every inter-peptide backbone H-bond that links
    position i on one peptide with position j on the other, whichever peptide
    donates (both donor/acceptor directions and both peptide assignments are
    summed). Off-diagonal bonds are written to (i, j) and (j, i) so the matrix
    is symmetric; the total is counted once per bond.
    """
    m = np.zeros((4, 4))
    total, skipped = 0.0, 0
    for _, r in hb.iterrows():
        a, d = res_to_pos(atom_res(r.acceptor)), res_to_pos(atom_res(r.donor))
        if a is None or d is None or a[0] == d[0] or a[1] > 4 or d[1] > 4:
            skipped += 1
            continue
        i, j = a[1] - 1, d[1] - 1
        m[i, j] += r.frac
        if i != j:
            m[j, i] += r.frac
        total += r.frac
    return m, total, skipped


def swatch_title(ax, system, text: str | None = None, y: float = 1.03, size: float = 9):
    """Panel title in ink with a small square in the system colour before it."""
    ax.text(0.0, y, "■", transform=ax.transAxes, ha="left", va="bottom",
            color=system.color, fontsize=size + 1, clip_on=False)
    ax.text(0.0, y, text or system.label, ha="left", va="bottom", fontsize=size,
            color=style.INK, clip_on=False,
            transform=offset_copy(ax.transAxes, fig=ax.figure, x=size + 3, y=0, units="points"))


def half_handles():
    return [Line2D([], [], ls="", marker="o", ms=7, mfc=style.SURFACE, mec=style.INK_2, mew=1.5,
                   label=f"{HALF_NAME['first']} ({HALF_FRAMES['first']})"),
            Line2D([], [], ls="", marker="o", ms=6, mfc=style.INK_2, mec=style.SURFACE, mew=0.8,
                   label=f"{HALF_NAME['second']} ({HALF_FRAMES['second']})")]


def dumbbell(ax, v1, v2, systems):
    """Half 1 open marker, half 2 filled, joined by a gray hairline."""
    for k, s in enumerate(systems):
        ax.plot([v1[k], v2[k]], [k, k], color=style.CONTEXT_GRAY, lw=style.LINE_W,
                zorder=1, solid_capstyle="butt")
        ax.plot(v1[k], k, "o", ms=8, mfc=style.SURFACE, mec=s.color, mew=1.6, zorder=3)
        ax.plot(v2[k], k, "o", ms=6.5, mfc=s.color, mec=style.SURFACE, mew=0.8, zorder=4)
    ax.set_yticks(range(len(systems)), [s.label for s in systems])
    ax.set_ylim(len(systems) - 0.5, -0.6)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def pad_range(lo, hi, frac=0.08):
    span = hi - lo
    return lo - frac * span, hi + frac * span


def text_on(value, vmax, thresh=0.55):
    return "white" if abs(value) > thresh * vmax else style.INK


# ====================================================================== loading
def load_all() -> dict:
    data = {}
    for s in SYSTEMS:
        d = halved_dir(s)
        per = {}
        for h in HALVES:
            ang = load_cpptraj_dat(d / f"pep1pep2_angle_{h}_half.dat").iloc[:, 1].to_numpy(np.float64)
            e2e = load_cpptraj_dat(d / f"end_to_end_dist_{h}_half.dat").iloc[:, 1].to_numpy(np.float64)
            hb = load_hbonds(d / f"interhbond_{h}_half.txt")
            mat, total, skipped = hbond_matrix(hb)
            implied = float(np.median(hb.frames / hb.frac)) if len(hb) else float("nan")
            per[h] = dict(angle=ang, e2e=e2e, hb=hb, hbmat=mat, hbtotal=total,
                          n_contact=len(ang), n_all=len(e2e), implied=implied, skipped=skipped)
            # QC: hbond Frac denominators must be the contact frames
            if np.isfinite(implied) and abs(implied - len(ang)) / len(ang) > 0.01:
                print(f"WARNING {s.key} {h}: hbond Frac implies {implied:.0f} frames, "
                      f"angle file has {len(ang)} contact frames", file=sys.stderr)
            if skipped:
                print(f"note {s.key} {h}: {skipped} hbond rows outside inter-peptide LIVA", file=sys.stderr)
        data[s.key] = per
    return data


def load_dihedrals(system) -> dict[str, pd.DataFrame]:
    """phi/psi of the contact frames, float32, read-only from the source folder."""
    out = {}
    for h in HALVES:
        path = system.analysis_dir / "halved" / f"multidihedral_{h}_half.dat"
        with open(path) as fh:
            header = fh.readline().lstrip("#").split()
        cols = [c for c in header if c != "Frame"]
        out[h] = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=header,
                             usecols=cols, dtype=np.float32, engine="c")
    return out


def phipsi(df: pd.DataFrame, pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Both peptides pooled: residue pos+1 (peptide 1) and pos+9 (peptide 2)."""
    r1, r2 = pos + 1, pos + 9
    phi = np.concatenate([df[f"phi:{r1}"].to_numpy(), df[f"phi:{r2}"].to_numpy()])
    psi = np.concatenate([df[f"psi:{r1}"].to_numpy(), df[f"psi:{r2}"].to_numpy()])
    return phi, psi


def classify(phi, psi) -> dict[str, float]:
    beta = (phi < -90) & ((psi >= 90) | (psi < -150))
    ppii = (phi >= -90) & (phi < -45) & (psi >= 90)
    alpha = (phi >= -160) & (phi < -20) & (psi >= -120) & (psi < 50)
    n = len(phi)
    f = {"β": beta.sum() / n, "PPII": ppii.sum() / n, "αR": alpha.sum() / n}
    f["other"] = 1.0 - sum(f.values())
    return f


# ====================================================================== figures
def fig_contact_frames(data):
    n1 = np.array([data[s.key]["first"]["n_contact"] for s in SYSTEMS])
    n2 = np.array([data[s.key]["second"]["n_contact"] for s in SYSTEMS])
    a1 = np.array([data[s.key]["first"]["n_all"] for s in SYSTEMS])
    a2 = np.array([data[s.key]["second"]["n_all"] for s in SYSTEMS])
    p1, p2 = 100 * n1 / a1, 100 * n2 / a2

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    dumbbell(ax, p1, p2, SYSTEMS)
    lo, hi = pad_range(min(p1.min(), p2.min()), max(p1.max(), p2.max()), 0.12)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("LIVA–LIVA contact frames (% of the 270 000 frames in each half)")
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.03, -0.7, "contact frames\nhalf 1 | half 2", transform=tr, ha="left", va="bottom",
            fontsize=7.5, color=style.MUTED)
    for k in range(len(SYSTEMS)):
        ax.text(1.03, k, f"{n1[k]:,} | {n2[k]:,}", transform=tr, ha="left", va="center",
                fontsize=8, color=style.INK_2)
    ax.legend(handles=half_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
              handletextpad=0.3, columnspacing=1.5)
    ax.set_title("How often LIVA segments are in contact (centre-of-mass distance < 7 Å), per half",
                 loc="left", fontsize=10)
    style.save(
        fig, "halved/contact_frames.png",
        "LIVA–LIVA contact frames per trajectory half",
        "Percentage (and number, right column) of the 270 000 frames of each trajectory half in which the "
        "centre-of-mass distance between the LIVA segments of the two peptides (cpptraj distance :2-5 vs "
        ":10-13, mass-weighted) is below 7 Å; these are the frames used for the angle, H-bond and "
        "backbone-dihedral analyses. Open marker = half 1 "
        "(frames 30 001–300 000), filled = half 2 (330 001–600 000); a long gray connector means the "
        "two halves disagree, i.e. contact formation is not converged.",
        GROUP, ALL_KEYS, 0)


def hbond_panels(data, which: str):
    """which = 'mean' (mean of halves) or 'diff' (half 2 - half 1)."""
    mats = {}
    for s in SYSTEMS:
        m1, m2 = data[s.key]["first"]["hbmat"], data[s.key]["second"]["hbmat"]
        mats[s.key] = 0.5 * (m1 + m2) if which == "mean" else m2 - m1
    if which == "mean":
        vmin, vmax, cmap = 0.0, max(m.max() for m in mats.values()), style.CMAP_SEQ
        fmt = lambda v: "0" if v == 0 else f"{v:.2f}".lstrip("0")  # noqa: E731
    else:
        vmax = max(np.abs(m).max() for m in mats.values())
        vmin, cmap = -vmax, style.CMAP_DIV
        fmt = lambda v: "0" if abs(v) < 0.005 else f"{v:+.2f}".replace("0.", ".").replace("-", "−")  # noqa: E731

    fig, axes = plt.subplots(2, 4, figsize=(10.2, 5.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.1, wspace=0.12, hspace=0.32)
    labels = POS_LABELS[:4]
    for ax, s in zip(axes.flat, SYSTEMS):
        m = mats[s.key]
        im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(range(4), labels)
        ax.set_yticks(range(4), labels)
        ax.tick_params(length=0)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, fmt(m[i, j]), ha="center", va="center", fontsize=7.5,
                        color=text_on(m[i, j], vmax))
        sub = (f"Σ = {np.triu(m).sum():.2f}" if which == "mean"
               else f"ΣΔ = {np.triu(m).sum():+.2f}".replace("-", "−"))
        swatch_title(ax, s)
        ax.text(1.0, 1.03, sub, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                color=style.INK_2)
    for ax in axes[1]:
        ax.set_xlabel("position on the other peptide")
    for ax in axes[:, 0]:
        ax.set_ylabel("position on one peptide")
    cax = fig.add_axes([0.905, 0.18, 0.015, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    if which == "mean":
        cb.set_label("H-bond occupancy (bonds per contact frame)")
        fig.suptitle("Inter-peptide backbone H-bonds between LIVA positions (mean of the two halves)",
                     x=0.08, ha="left", fontsize=11)
        style.save(
            fig, "halved/hbond_maps.png",
            "Inter-peptide backbone H-bond occupancy maps (mean of halves)",
            "Each 4×4 map gives, for residue i of one peptide and residue j of the other (LIVA only; the "
            "cpptraj hbond mask was :2-5,10-13, nointramol), the summed fraction of contact frames in "
            "which an N–H···O=C H-bond links them (cpptraj default criteria: N···O < 3.0 Å, N–H···O angle "
            "> 135°), summing both donor/acceptor directions and both "
            "peptide assignments, so the map is symmetric and an off-diagonal bond appears in both "
            "(i,j) and (j,i). Values are the mean of half 1 and half 2 and are per LIVA–LIVA contact "
            "frame (the frames the H-bond analysis ran on); Σ = upper triangle + diagonal = mean number "
            "of LIVA–LIVA backbone H-bonds per contact frame. One colour scale for all systems.",
            GROUP, ALL_KEYS, 1)
    else:
        cb.set_label("Δ occupancy, half 2 − half 1 (bonds per contact frame)")
        fig.suptitle("Change in inter-peptide backbone H-bond occupancy between trajectory halves",
                     x=0.08, ha="left", fontsize=11)
        style.save(
            fig, "halved/hbond_halves_diff.png",
            "H-bond occupancy: second half minus first half",
            "Same symmetric LIVA×LIVA occupancy matrices as the H-bond maps, shown as half 2 (frames "
            "330 001–600 000) minus half 1 (frames 30 001–300 000), per contact frame. Red = more "
            "frequent in the second half, blue = less frequent, gray ≈ converged; ΣΔ is the change in "
            "the mean number of LIVA–LIVA backbone H-bonds per contact frame. One symmetric colour "
            "scale for all systems.",
            GROUP, ALL_KEYS, 2)


def fig_hbond_total(data):
    t1 = np.array([data[s.key]["first"]["hbtotal"] for s in SYSTEMS])
    t2 = np.array([data[s.key]["second"]["hbtotal"] for s in SYSTEMS])
    c1 = np.array([data[s.key]["first"]["n_contact"] / data[s.key]["first"]["n_all"] for s in SYSTEMS])
    c2 = np.array([data[s.key]["second"]["n_contact"] / data[s.key]["second"]["n_all"] for s in SYSTEMS])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), sharey=True)
    fig.subplots_adjust(wspace=0.12, left=0.15, right=0.98, top=0.84, bottom=0.25)
    panels = [(axes[0], t1, t2, "H-bonds per LIVA–LIVA contact frame",
               "a  per contact frame (as reported by cpptraj)"),
              (axes[1], t1 * c1, t2 * c2, "H-bonds per frame, all 270 000 frames (lower bound)",
               "b  per frame of the whole half (= a × contact fraction)")]
    for ax, v1, v2, xl, ttl in panels:
        dumbbell(ax, v1, v2, SYSTEMS)
        lo, hi = pad_range(min(v1.min(), v2.min()), max(v1.max(), v2.max()), 0.12)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(xl)
        ax.set_title(ttl, loc="left", fontsize=9.5)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].spines["left"].set_visible(False)
    fig.legend(handles=half_handles(), loc="lower center", bbox_to_anchor=(0.56, 0.01), ncol=2,
               handletextpad=0.3, columnspacing=1.5)
    fig.suptitle("Mean number of inter-peptide LIVA–LIVA backbone H-bonds, per trajectory half",
                 x=0.15, ha="left", fontsize=11)
    style.save(
        fig, "halved/hbond_total.png",
        "Total inter-peptide backbone H-bonds per half",
        "Sum of the cpptraj Frac column over all inter-peptide LIVA–LIVA backbone H-bonds = mean number "
        "of such H-bonds per frame. (a) per LIVA–LIVA contact frame (the frames hbond was run on); "
        "(b) the same multiplied by the fraction of contact frames in the half, i.e. per frame of the "
        "whole half; this is a lower bound, because hbond was never run on frames whose LIVA–LIVA "
        "centre-of-mass distance is ≥ 7 Å, and staggered/end-on contacts there can still carry H-bonds. "
        "Open = half 1, filled = half 2.",
        GROUP, ALL_KEYS, 3)


def dist_figure(data, key, bins, xlabel, dens_unit, title, caption, relpath, order,
                xlim, xticks=None, isotropic=False):
    centers = 0.5 * (bins[1:] + bins[:-1])
    dens = {}
    for s in SYSTEMS:
        h1 = np.histogram(data[s.key]["first"][key], bins=bins, density=True)[0]
        h2 = np.histogram(data[s.key]["second"][key], bins=bins, density=True)[0]
        dens[s.key] = (h1, h2)
    ymax = 1.08 * max(max(h1.max(), h2.max()) for h1, h2 in dens.values())

    fig = plt.figure(figsize=(14.0, 6.0))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.15, 2.0], wspace=0.15,
                             left=0.05, right=0.99, top=0.83, bottom=0.1)
    ax0 = fig.add_subplot(outer[0])
    inner = outer[1].subgridspec(2, 4, wspace=0.1, hspace=0.42)
    small = []
    for r in range(2):
        for c in range(4):
            small.append(fig.add_subplot(inner[r, c], sharex=small[0] if small else None,
                                         sharey=small[0] if small else None))

    # left: overlay of the equal-weight mean of the two half densities. For end-to-end
    # (270 000 frames per half) this equals the pooled density; for the angle the
    # halves have different numbers of contact frames, so it is NOT the pooled density.
    if isotropic:
        iso = np.sin(np.deg2rad(centers)) * np.pi / 360.0  # sin(θ)/2 per rad -> per degree
        ax0.plot(centers, iso, color=style.MUTED, lw=style.THIN_W, ls=(0, (4, 3)),
                 label="isotropic, (π/360)·sin θ", zorder=1)
    for s in SYSTEMS:
        h1, h2 = dens[s.key]
        ax0.plot(centers, 0.5 * (h1 + h2), color=s.color, lw=style.LINE_W, label=s.label, zorder=2)
    ax0.set_xlim(*xlim)
    ax0.set_ylim(0, ymax)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(f"probability density ({dens_unit})")
    ax0.set_title("all systems, mean of the two halves", loc="left", fontsize=9.5)
    if xticks is not None:
        ax0.set_xticks(xticks)
    ax0.legend(loc="upper left", ncol=1, handlelength=1.6, fontsize=7.5, borderaxespad=0.3)

    # right: per system, half 1 (system colour) vs half 2 (gray)
    for k, (ax, s) in enumerate(zip(small, SYSTEMS)):
        h1, h2 = dens[s.key]
        if isotropic:
            ax.plot(centers, iso, color=style.MUTED, lw=0.6, ls=(0, (4, 3)), zorder=1)
        ax.plot(centers, h2, color=style.CONTEXT_GRAY, lw=2.2, zorder=2)
        ax.plot(centers, h1, color=s.color, lw=style.THIN_W + 0.2, zorder=3)
        swatch_title(ax, s, size=8.5)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, ymax)
        if xticks is not None:
            ax.set_xticks(xticks)
        if k % 4:
            ax.tick_params(axis="y", labelleft=False)
        else:
            ax.set_ylabel(f"density ({dens_unit})", fontsize=8, labelpad=2)
        if k < 4:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel(xlabel, fontsize=8)
    handles = [Line2D([], [], color=style.INK_2, lw=style.THIN_W + 0.2,
                      label=f"half 1 ({HALF_FRAMES['first']}), in system colour"),
               Line2D([], [], color=style.CONTEXT_GRAY, lw=2.2,
                      label=f"half 2 ({HALF_FRAMES['second']})")]
    if isotropic:
        handles.append(Line2D([], [], color=style.MUTED, lw=0.6, ls=(0, (4, 3)), label="isotropic"))
    pos = small[0].get_position()
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(pos.x0, 0.885), ncol=3,
               fontsize=8, handlelength=2.2, columnspacing=1.6, borderaxespad=0)
    fig.suptitle(title, x=0.05, ha="left", y=0.975, fontsize=11)
    style.save(fig, relpath, title, caption, GROUP, ALL_KEYS, order, dpi=150)
    return dens


def fig_end_to_end(data):
    bins = np.arange(3.0, 19.0 + 1e-9, 0.2)
    dist_figure(
        data, "e2e", bins, "end-to-end distance (Å)", "Å⁻¹",
        "End-to-end distance of peptide 1 (Leu Cα – residue-X Cα), half 1 vs half 2",
        "Distribution of the :2@CA–:7@CA distance (Cα of L1 to Cα of X6) of peptide 1 over all "
        "270 000 frames of each half (not only contact frames), 0.2 Å bins, normalised densities. "
        "Left: all systems overlaid, mean of the two half densities (equal-size halves, so identical to "
        "pooling them); right: one panel per system, half 1 in "
        "the system colour drawn over half 2 in gray, so a visible gray rim = the halves agree. "
        "Shared axes everywhere. For LIVAGE (E⁻/E⁰) peptide 1 is the charged (GLU) one.",
        "halved/end_to_end.png", 4, xlim=(3, 19), xticks=np.arange(4, 19, 2))


def fig_angle(data):
    bins = np.arange(0.0, 180.0 + 1e-9, 3.0)
    dist_figure(
        data, "angle", bins, "pep1–pep2 vector angle (°)", "deg⁻¹",
        "Angle between the peptide-1 and peptide-2 backbone vectors (LIVA–LIVA contact frames)",
        "Distribution of the angle between the residue-centre-of-mass vectors v1 = :2→:5 (L1→A4 of "
        "peptide 1) and v2 of peptide 2, over LIVA–LIVA contact frames only (LIVA centre-of-mass "
        "distance < 7 Å), 3° bins, normalised densities; 0° = parallel, 180° = antiparallel, and the "
        "dashed gray curve is the isotropic (random-orientation) expectation, sin θ/2 per radian = "
        "(π/360)·sin θ per degree. Left: all systems, equal-weight mean of the two half densities (the "
        "halves have different numbers of contact frames); right: half 1 in system colour over half 2 "
        "in gray, shared axes. " + ANGLE_CAVEAT,
        "halved/pep_angle.png", 5, xlim=(0, 180), xticks=np.arange(0, 181, 30), isotropic=True)


def compute_dihedrals():
    """Per system: pooled 2D histograms per position + SS fractions per half."""
    edges = np.arange(-180.0, 180.0 + 1e-9, 5.0)
    hists, rows = {}, []
    for s in SYSTEMS:
        dfs = load_dihedrals(s)
        for pos in range(1, 7):
            res = s.sequence[pos - 1]
            pooled_phi, pooled_psi = [], []
            for h in HALVES:
                phi, psi = phipsi(dfs[h], pos)
                pooled_phi.append(phi)
                pooled_psi.append(psi)
                rows.append(dict(system=s.key, label=s.label, position=pos, residue=res, half=h,
                                 n_samples=len(phi), **classify(phi, psi)))
            phi, psi = np.concatenate(pooled_phi), np.concatenate(pooled_psi)
            rows.append(dict(system=s.key, label=s.label, position=pos, residue=res, half="pooled",
                             n_samples=len(phi), **classify(phi, psi)))
            H, _, _ = np.histogram2d(phi, psi, bins=[edges, edges])
            hists[(s.key, pos)] = H / (H.sum() * 25.0)  # density per deg^2
        print(f"  dihedrals read: {s.key} ({len(dfs['first'])} + {len(dfs['second'])} frames)")
        del dfs
    ss = pd.DataFrame(rows).rename(columns={"β": "beta", "PPII": "ppii", "αR": "alpha_r"})
    return edges, hists, ss


def draw_regions(ax):
    kw = dict(color=style.INK_2, lw=0.45, alpha=0.75, zorder=4)
    for x0, x1, y0, y1 in [(-180, -90, 90, 180), (-180, -90, -180, -150),  # beta
                           (-90, -45, 90, 180),                              # PPII
                           (-160, -20, -120, 50)]:                           # alpha-R
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], **kw)


def fig_ramachandran(edges, hists):
    vals = np.concatenate([h[h > 0] for h in hists.values()])
    vmax = vals.max()
    vmin = max(vals.min(), vmax * 1e-3)  # three decades
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = style.CMAP_SEQ.copy()
    cmap.set_bad(style.SURFACE)
    cmap.set_under(style.SEQ_BLUE[0])

    nrow, ncol = len(SYSTEMS), 6
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.4, 14.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.16, right=0.9, top=0.93, bottom=0.05, wspace=0.08, hspace=0.1)
    ticks = [-180, -90, 0, 90, 180]
    ticklabels = ["−180", "", "0", "", "180"]
    for r, s in enumerate(SYSTEMS):
        for c in range(ncol):
            ax = axes[r, c]
            pos = c + 1
            H = np.ma.masked_equal(hists[(s.key, pos)].T, 0)  # rows = psi
            mesh = ax.pcolormesh(edges, edges, H, cmap=cmap, norm=norm, shading="flat",
                                 rasterized=True)
            ax.grid(False)
            draw_regions(ax)
            ax.set_xlim(-180, 180)
            ax.set_ylim(-180, 180)
            ax.set_aspect("equal")
            ax.set_xticks(ticks, ticklabels)
            ax.set_yticks(ticks, ticklabels)
            ax.tick_params(labelsize=7, length=2)
            for side in ("top", "right"):
                ax.spines[side].set_visible(True)
            if pos == 6:
                ax.text(0.97, 0.04, s.x_resname, transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=6.5, color=style.INK_2,
                        bbox=dict(boxstyle="round,pad=0.15", fc=style.SURFACE, ec="none", alpha=0.85))
            if r == 0:
                ax.set_title(POS_LABELS[c], fontsize=10)
            if r == nrow - 1:
                ax.set_xlabel("φ (°)")
            if c == 0:
                ax.set_ylabel("ψ (°)", labelpad=1)
                ax.annotate(s.label.replace(" (", "\n("), xy=(0, 0.5), xycoords="axes fraction",
                            xytext=(-52, 0), textcoords="offset points", ha="right", va="center",
                            fontsize=9, color=style.INK)
                ax.annotate("■", xy=(0, 0.5), xycoords="axes fraction", xytext=(-38, 0),
                            textcoords="offset points", ha="right", va="center", fontsize=10,
                            color=s.color)
    for ax in axes[-1]:  # keep the ±180 labels inside their own panel
        labs = ax.get_xticklabels()
        labs[0].set_horizontalalignment("left")
        labs[-1].set_horizontalalignment("right")
    cax = fig.add_axes([0.915, 0.3, 0.014, 0.4])
    cb = fig.colorbar(mesh, cax=cax, extend="min")
    cb.outline.set_visible(False)
    cb.set_label("probability density (deg⁻², log scale)")
    fig.suptitle("Backbone φ/ψ of each position during LIVA–LIVA contact (both peptides, both halves pooled)",
                 x=0.16, ha="left", y=0.965, fontsize=11)
    style.save(
        fig, "halved/ramachandran_grid.png",
        "Ramachandran maps per system and position (contact frames)",
        "Rows = systems, columns = peptide positions L1…X6 (residue name of X6 at the bottom right; "
        "LIVAGE (E⁻/E⁰) pools GLU of peptide 1 and GLH of peptide 2). Each panel is a 5°×5° histogram "
        "of the cpptraj multidihedral φ/ψ (ACE and NHE caps make φ and ψ defined for every position), "
        "pooling both peptides and both halves, restricted to LIVA–LIVA contact frames (LIVA "
        "centre-of-mass distance < 7 Å), no "
        "downsampling; colour = probability density on one shared log scale, empty bins blank. Thin "
        "boxes mark the β, PPII and αR regions used in the secondary-structure figure.",
        GROUP, ALL_KEYS, 6, dpi=130)


def fig_ss(ss: pd.DataFrame):
    pooled = ss[ss.half == "pooled"]
    cols = [("beta", "β"), ("ppii", "PPII"), ("alpha_r", "αR"), ("other", "other")]
    mats = {}
    for col, _ in cols:
        mats[col] = np.array([[pooled[(pooled.system == s.key) & (pooled.position == p)][col].iloc[0]
                               for p in range(1, 7)] for s in SYSTEMS]) * 100
    vmax = max(m.max() for m in mats.values())
    h1 = ss[ss.half == "first"].set_index(["system", "position"])
    h2 = ss[ss.half == "second"].set_index(["system", "position"])
    maxdiff = 100 * max((h1[c] - h2[c]).abs().max() for c, _ in cols)

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.2), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.92, top=0.82, bottom=0.13, wspace=0.06)
    for ax, (col, name) in zip(axes, cols):
        m = mats[col]
        im = ax.imshow(m, cmap=style.CMAP_SEQ, vmin=0, vmax=vmax, aspect="auto")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color=text_on(m[i, j], vmax))
        ax.set_xticks(range(6), POS_LABELS)
        ax.tick_params(length=0)
        ax.set_title(f"{name}", loc="left", fontsize=10, pad=15)
        ax.text(0, 1.015, SS_REGIONS[name], transform=ax.transAxes, ha="left", va="bottom",
                fontsize=6.8, color=style.INK_2)
        ax.set_xlabel("position")
    axes[0].set_yticks(range(len(SYSTEMS)), [s.label for s in SYSTEMS])
    tr = blended_transform_factory(axes[0].transAxes, axes[0].transData)
    for k, s in enumerate(SYSTEMS):
        axes[0].text(-0.01, k, "■", transform=tr, ha="right", va="center", fontsize=8, color=s.color)
    axes[0].tick_params(axis="y", pad=12)
    for t in axes[0].get_yticklabels():
        t.set_color(style.INK_2)
    cax = fig.add_axes([0.93, 0.15, 0.012, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label("population (% of residue conformations)")
    fig.suptitle("Backbone conformation populations during LIVA–LIVA contact (both peptides, both halves pooled)",
                 x=0.1, ha="left", y=0.98, fontsize=11)
    caption = (
        "Percentage of φ/ψ samples of each position (both peptides and both halves pooled, LIVA–LIVA "
        "contact frames, LIVA centre-of-mass distance < 7 Å) that fall in each region; the four values "
        "of a given system and position (same cell in the four panels) sum to 100 %. "
        "Regions (degrees, simple rectangular definitions after common Ramachandran partitions): "
        "β: φ ∈ [−180, −90), ψ ∈ [90, 180] ∪ [−180, −150); PPII: φ ∈ [−90, −45), ψ ∈ [90, 180]; "
        "αR: φ ∈ [−160, −20), ψ ∈ [−120, 50); other = everything else (αL, bridge, etc.). "
        f"Half 1 and half 2 differ by at most {maxdiff:.1f} percentage points in any cell "
        "(per-half values in secondary_structure_populations.csv). One colour scale for all panels.")
    style.save(fig, "halved/secondary_structure_populations.png",
               "Secondary-structure region populations per position", caption, GROUP, ALL_KEYS, 7)
    return maxdiff


# ====================================================================== tables
def write_tables(data, ss: pd.DataFrame | None):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, hrows = [], []
    for s in SYSTEMS:
        for h in HALVES:
            d = data[s.key][h]
            frac = d["n_contact"] / d["n_all"]
            rows.append(dict(
                system=s.key, half=h, contact_frames=d["n_contact"],
                contact_pct=round(100 * frac, 3),
                total_hbond_frac=round(d["hbtotal"], 4),
                e2e_mean_A=round(float(d["e2e"].mean()), 3),
                angle_mean_deg=round(float(d["angle"].mean()), 2),
                label=s.label, frames_in_half=d["n_all"],
                # lower bound: hbond only ran on LIVA COM-distance < 7 A frames
                hbonds_per_all_frames_lower_bound=round(d["hbtotal"] * frac, 4),
                e2e_median_A=round(float(np.median(d["e2e"])), 3),
                angle_median_deg=round(float(np.median(d["angle"])), 2)))
            m = d["hbmat"]
            for i in range(4):
                for j in range(4):
                    hrows.append(dict(system=s.key, half=h, pos_i=i + 1, res_i=POS_LABELS[i],
                                      pos_j=j + 1, res_j=POS_LABELS[j],
                                      occupancy_per_contact_frame=round(m[i, j], 4)))
    summary = pd.DataFrame(rows)
    summary.to_csv(assert_inside_overall(OUT / "halved_summary.csv"), index=False)
    pd.DataFrame(hrows).to_csv(assert_inside_overall(OUT / "hbond_occupancy.csv"), index=False)
    if ss is not None:
        out = ss.copy()
        for c in ("beta", "ppii", "alpha_r", "other"):
            out[c] = out[c].round(4)
        out.to_csv(assert_inside_overall(OUT / "secondary_structure_populations.csv"), index=False)
    return summary


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-dihedrals", action="store_true",
                    help="skip the Ramachandran / secondary-structure figures (large source reads)")
    args = ap.parse_args()

    style.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading halved data ...")
    data = load_all()
    fig_contact_frames(data)
    hbond_panels(data, "mean")
    hbond_panels(data, "diff")
    fig_hbond_total(data)
    fig_end_to_end(data)
    fig_angle(data)
    ss = None
    if not args.no_dihedrals:
        print("reading multidihedral files from the source folders (read-only) ...")
        edges, hists, ss = compute_dihedrals()
        fig_ramachandran(edges, hists)
        fig_ss(ss)
    summary = write_tables(data, ss)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.iloc[:, :7].to_string(index=False))
    print(f"wrote figures and tables to {OUT}")


if __name__ == "__main__":
    main()
