"""Angle x end-to-end 2D PMFs of the LIVA-LIVA contact frames.

Shared by fig_e2e_angle_2d.py (stand-alone figures, tables, QC) and
fig_system_rows.py (three extra columns of the big per-system figure).

Inputs, per system, from halved/ (snapshot copy preferred, see common.halved_dir):

  pep1pep2_angle_first_half.dat   angle between v1 = :2 -> :5 and v2 = :9 -> :12,
  pep1pep2_angle_second_half.dat  one row per contact frame (LIVA-LIVA COM <= 7 A)
  end_to_end_dist_pepA.dat        SECTION 3 of the halved cpptraj input:
  end_to_end_dist_pepB.dat        pep A = :2@CA - :7@CA   (peptide 1)
                                  pep B = :10@CA - :15@CA (peptide 2)
                                  on filtered_frames_first_half.nc followed by
                                  filtered_frames_second_half.nc

Row k of [angle first half; angle second half] is the same frame as row k of
both end-to-end files: load() checks the counts, and on 2026-11-04 pep A on
these frames was checked to equal end_to_end_dist_{first,second}_half.dat
(all frames) at exactly the frames passing the LIVA-LIVA filter.

Surfaces (one 4.5 deg x 0.4 A grid for every system):

  A    F(theta, d_A) = -kT ln P(theta, d_A)
  B    F(theta, d_B) = -kT ln P(theta, d_B)
  avg  (F_A + F_B) / 2 bin by bin, empty where either is empty

P = bin count / number of contact frames with both halves pooled, kT at 300 K
(constants of pmf_from_cpptraj.py), each surface shifted so its lowest bin is 0,
no Jacobian (sin theta or d^2; the 1D PMFs and angle histograms have none either).

Differences between surfaces (half 1 vs half 2, A vs B) are taken on the
unshifted -kT ln p (raw_surfaces), each normalised over its own frames, so the
arbitrary per-surface zero does not enter them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

import style
from common import System, halved_dir, load_cpptraj_dat

K_B = 0.0019872041  # kcal/mol/K, as in pmf_from_cpptraj.py
TEMPERATURE = 300.0  # K
KT = K_B * TEMPERATURE

# 4.5 deg = the bin width of the original angle histograms. Rounded so every edge is the
# float nearest its decimal value (linspace alone gives 5.800000000000001 etc., which would
# put a 4-decimal sample of exactly 5.8 one bin low).
ANG_EDGES = np.round(np.linspace(0.0, 180.0, 41), 6)  # deg, 4.5 deg bins
E2E_EDGES = np.round(np.linspace(3.0, 19.0, 41), 6)  # A, 0.4 A bins
ANG_BIN = float(ANG_EDGES[1] - ANG_EDGES[0])
E2E_BIN = float(round(E2E_EDGES[1] - E2E_EDGES[0], 6))
ANG_C = 0.5 * (ANG_EDGES[1:] + ANG_EDGES[:-1])
E2E_C = 0.5 * (E2E_EDGES[1:] + E2E_EDGES[:-1])
FMAX = 4.0  # colour cap (kcal/mol); higher bins and empty bins are drawn as the lightest step
LEVELS = (1.0, 2.0)  # contour levels, kcal/mol (a 3 kcal/mol line mostly traces noise at the sampling edge)
NOTE_XY = (0.97, 0.03)  # panel-corner note at the bottom right: high angle + compact peptide is never visited

TYPES = ("A", "B", "avg")
TYPE_TITLE = {"A": "pep A (peptide 1)", "B": "pep B (peptide 2)", "avg": "average of A and B"}
TYPE_MASK = {"A": ":2@CA–:7@CA", "B": ":10@CA–:15@CA", "avg": "(F_A + F_B)/2"}
CMAP_LIGHT = (0.98, 0.98, 0.975)
SECTION3_FILES = ("end_to_end_dist_pepA.dat", "end_to_end_dist_pepB.dat")


@dataclass
class ContactFrames:
    """Per-frame data of the contact frames, half 1 followed by half 2."""
    angle: np.ndarray  # deg
    eA: np.ndarray  # A, peptide 1 end-to-end
    eB: np.ndarray  # A, peptide 2 end-to-end
    n1: int
    n2: int

    @property
    def n(self) -> int:
        return self.n1 + self.n2

    def halves(self) -> tuple[slice, slice]:
        return slice(0, self.n1), slice(self.n1, self.n)


def _col(path) -> np.ndarray:
    return load_cpptraj_dat(path).iloc[:, 1].to_numpy(np.float64)


def load(system: System) -> ContactFrames:
    d = halved_dir(system)
    missing = [f for f in SECTION3_FILES if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{system.key}: {', '.join(missing)} not in {d}. Run SECTION 3 of the halved cpptraj input "
            "(end-to-end of pep A :2@CA :7@CA and pep B :10@CA :15@CA on the filtered frames), then "
            "scripts/sync_data.py (or ./update.sh).")
    a1 = _col(d / "pep1pep2_angle_first_half.dat")
    a2 = _col(d / "pep1pep2_angle_second_half.dat")
    eA = _col(d / SECTION3_FILES[0])
    eB = _col(d / SECTION3_FILES[1])
    n1, n2 = len(a1), len(a2)
    if not len(eA) == len(eB) == n1 + n2:
        raise ValueError(f"{system.key}: angle rows {n1} + {n2} != end-to-end rows A {len(eA)} / B {len(eB)}; "
                         "re-run SECTION 3 of the halved cpptraj input")
    cf = ContactFrames(np.concatenate([a1, a2]), eA, eB, n1, n2)
    out = {name: int(((v < lo) | (v > hi)).sum()) for name, v, lo, hi in
           (("angle", cf.angle, ANG_EDGES[0], ANG_EDGES[-1]), ("pep A", eA, E2E_EDGES[0], E2E_EDGES[-1]),
            ("pep B", eB, E2E_EDGES[0], E2E_EDGES[-1]))}
    for name, k in out.items():
        if k:
            print(f"WARNING {system.key}: {k} {name} values outside the 2D PMF grid are not counted", file=sys.stderr)
    return cf


def raw_free_energy(theta: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """-kT ln p per bin (p = count / frames), NaN where empty; and the counts.
    Arrays are indexed [angle bin, end-to-end bin]."""
    h = np.histogram2d(theta, d, bins=(ANG_EDGES, E2E_EDGES))[0]
    with np.errstate(divide="ignore"):
        f = -KT * np.log(h / len(theta))
    f[h == 0] = np.nan
    return f, h


def raw_surfaces(cf: ContactFrames, sl: slice = slice(None)) -> tuple[dict, dict]:
    """{'A', 'B', 'avg'} -> unshifted -kT ln p over the frames in sl (NaN = empty), and counts.
    A and B share the frame total, so (F_A + F_B)/2 and F_A - F_B compare like with like."""
    fA, hA = raw_free_energy(cf.angle[sl], cf.eA[sl])
    fB, hB = raw_free_energy(cf.angle[sl], cf.eB[sl])
    return {"A": fA, "B": fB, "avg": 0.5 * (fA + fB)}, {"A": hA, "B": hB}


def _zero_min(f: np.ndarray) -> np.ndarray:
    return f - np.nanmin(f)


def surfaces(cf: ContactFrames, sl: slice = slice(None)) -> tuple[dict, dict, dict]:
    """(shifted, counts, raw): shifted = {'A', 'B', 'avg'} -> F with min 0 (what is plotted),
    for the frames in sl (default: all contact frames, both halves pooled)."""
    raw, counts = raw_surfaces(cf, sl)
    return {t: _zero_min(f) for t, f in raw.items()}, counts, raw


def half_surfaces(cf: ContactFrames) -> tuple[tuple[dict, dict], tuple[dict, dict]]:
    """((shifted half 1, raw half 1), (shifted half 2, raw half 2)); each half normalised over its own frames."""
    out = []
    for sl in cf.halves():
        shifted, _, raw = surfaces(cf, sl)
        out.append((shifted, raw))
    return out[0], out[1]


def minimum(f: np.ndarray) -> tuple[float, float]:
    """(angle, end-to-end) bin centre of the lowest bin."""
    i, j = np.unravel_index(np.nanargmin(f), f.shape)
    return float(ANG_C[i]), float(E2E_C[j])


def min_gap(f: np.ndarray) -> float:
    """F of the second-lowest bin minus the lowest: how sharply the lowest bin is resolved."""
    v = np.sort(f[np.isfinite(f)])
    return float(v[1] - v[0]) if v.size > 1 else float("nan")


# ------------------------------------------------------------------ drawing
def pmf_cmap(system: System) -> LinearSegmentedColormap:
    """One-hue ramp in the system colour: dark at F = 0 (most probable) -> near white at FMAX.
    The same lightness steps as the H-bond occupancy ramp of fig_system_rows, reversed.
    Hues differ in lightness, so compare systems by the contour lines, not by shade."""
    c = np.array(to_rgb(system.color))
    cm = LinearSegmentedColormap.from_list(
        f"pmf2d_{system.key}", [tuple(0.55 * c), tuple(c), tuple(0.5 * c + 0.5), CMAP_LIGHT])
    cm.set_over(CMAP_LIGHT)
    cm.set_bad(CMAP_LIGHT)
    return cm


NORM = Normalize(0.0, FMAX)


def style_axes(ax, *, bottom: bool, left: bool, fs: float) -> None:
    ax.set_xlim(ANG_EDGES[0], ANG_EDGES[-1])
    ax.set_ylim(E2E_EDGES[0], E2E_EDGES[-1])
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_yticks([4, 8, 12, 16])
    ax.grid(False)
    ax.tick_params(labelsize=fs, length=2, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
        sp.set_color(style.AXIS)
    if bottom:
        ax.set_xlabel("pep1–pep2 angle (deg)", fontsize=fs + 0.5)
    else:
        ax.tick_params(labelbottom=False)
    if left:
        ax.set_ylabel("end-to-end Cα–Cα (Å)", fontsize=fs + 0.5)
    else:
        ax.tick_params(labelleft=False)


def draw(ax, system: System, f: np.ndarray, *, bottom: bool = True, left: bool = True,
         fs: float = 7.0) -> None:
    """Heat map of one surface + contours at LEVELS + a ringed dot on the lowest bin.
    The basin floor is flat (several bins within ~0.1 kcal/mol), so the dot is not labelled
    with a position; the bin centre is in pmf2d_summary.csv together with min_gap_kcal."""
    fm = np.ma.masked_invalid(f).T  # rows = end-to-end, cols = angle
    ax.pcolormesh(ANG_EDGES, E2E_EDGES, fm, cmap=pmf_cmap(system), norm=NORM, shading="flat",
                  rasterized=True, zorder=1)
    ax.contour(ANG_C, E2E_C, fm, levels=LEVELS, colors=style.INK, linewidths=0.4, alpha=0.75, zorder=2)
    th, dd = minimum(f)
    ax.plot([th], [dd], "o", ms=3.6, mfc=style.SURFACE, mec=style.INK, mew=0.8, zorder=3)
    style_axes(ax, bottom=bottom, left=left, fs=fs)


def colorbar(fig, cax, system: System, label: str, orientation: str = "horizontal", fs: float = 7.0):
    cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=pmf_cmap(system)), cax=cax,
                      orientation=orientation, extend="max")
    cb.set_ticks(np.arange(0, FMAX + 0.5, 1.0))
    cb.set_label(label, fontsize=fs)
    cb.ax.tick_params(labelsize=fs - 0.5, length=2)
    cb.outline.set_linewidth(0.4)
    # the contour levels, marked on the bar as on the map
    for v in LEVELS:
        (cb.ax.axvline if orientation == "horizontal" else cb.ax.axhline)(v, color=style.INK, lw=0.6, alpha=0.75)
    return cb
