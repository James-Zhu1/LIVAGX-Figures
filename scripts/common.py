"""Shared configuration and loaders for the LIVAGX overall documentation.

Every other script in overall/scripts imports from here, so the system list,
paths, colours and PMF conventions live in exactly one place.

Read-only rule: nothing in this package may write outside OVERALL. The source
analysis folders are only ever read.
"""

from __future__ import annotations

import fcntl
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths
OVERALL = Path(__file__).resolve().parents[1]
PROJECT = OVERALL.parent  # .../1-9/60mM_parallel  (READ ONLY)
DATA = OVERALL / "data"  # snapshot of the small analysis files
FIGURES = OVERALL / "figures"  # new figures made by overall/scripts
DOCS = OVERALL / "docs"  # generated HTML site
MANIFEST = DATA / "manifest.json"
FIGURE_INDEX = FIGURES / "index.json"

PYTHON = "/hopper/home/jzii2024/miniconda3/envs/pmf/bin/python"

# Files never copied into the snapshot (user rule) ...
EXCLUDE_SUFFIXES = (".in", ".prmtop", ".log", ".nc")
# ... and files too large for git: recorded in the manifest (size, mtime) only.
MAX_COPY_BYTES = 20 * 1024 * 1024


# ------------------------------------------------------------------ systems
@dataclass(frozen=True)
class System:
    key: str  # short id used for folder / file names
    label: str  # human label for legends
    sequence: str  # one-letter sequence, e.g. LIVAGE
    x_resname: str  # Amber residue name of residue 6 (from prmtop)
    note: str  # protonation / charge note
    analysis_dir: Path  # source analysis_continued folder (read only)
    color: str  # fixed categorical colour (entity, never rank)

    @property
    def x(self) -> str:
        return self.sequence[-1]

    @property
    def pmf_dir(self) -> Path:
        return self.analysis_dir / f"{self.sequence}_PMF"

    @property
    def snapshot(self) -> Path:
        return DATA / self.key

    @property
    def snapshot_pmf(self) -> Path:
        return self.snapshot / f"{self.sequence}_PMF"


# Categorical slots 1-8 of the validated reference palette (light mode), in
# fixed order. Exactly eight systems -> one slot each; a ninth system must not
# get a generated hue (facet or fold instead).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]

_TT = "_tripled_time_new_ff"
_SYS = [
    ("livage_cc", "LIVAGE (E⁻/E⁻)", "LIVAGE", "GLU/GLU",
     "both Glu charged", "livage/livage_charged_charged" + _TT + "/analysis_continued"),
    ("livage_cn", "LIVAGE (E⁻/E⁰)", "LIVAGE", "GLU/GLH",
     "one Glu charged, one neutral (GLH)", "livage/livage_charged_neutral" + _TT + "/analysis_continued"),
    ("livage_nn", "LIVAGE (E⁰/E⁰)", "LIVAGE", "GLH/GLH",
     "both Glu neutral (GLH)", "livage/livage_neutral_neutral" + _TT + "/analysis_continued"),
    ("livagg", "LIVAGG", "LIVAGG", "GLY", "neutral", "livagg/livagg" + _TT + "/analysis_continued"),
    ("livagh", "LIVAGH (H⁺)", "LIVAGH", "HIP", "His doubly protonated (HIP, +1)",
     "livagh/livagh" + _TT + "/analysis_continued"),
    ("livagk", "LIVAGK", "LIVAGK", "LYS", "Lys +1", "livagk/livagk" + _TT + "/analysis_continued"),
    ("livagq", "LIVAGQ", "LIVAGQ", "GLN", "neutral", "livagq/livagq" + _TT + "/analysis_continued"),
    # NB: the source folder really is spelled "analysis_contined".
    ("livagr", "LIVAGR", "LIVAGR", "ARG", "Arg +1", "livagr/livagr" + _TT + "/analysis_contined"),
]
SYSTEMS = [System(k, lab, seq, rn, note, PROJECT / rel, PALETTE[i])
           for i, (k, lab, seq, rn, note, rel) in enumerate(_SYS)]
BY_KEY = {s.key: s for s in SYSTEMS}

# Residue layout in the prmtop: ACE(1) L I V A G X(2-7) NHE(8) ACE(9) L I V A G X(10-15) NHE(16)
POSITIONS = list(range(1, 7))  # peptide positions 1..6; position 6 = X


def residue_letters(system: System) -> list[str]:
    return list(system.sequence)


# ------------------------------------------------------------------ PMF ids
def pair_ids() -> list[tuple[int, int]]:
    """36 ordered (pos on peptide 1, pos on peptide 2) pairs, row-major."""
    return [(i, j) for i in POSITIONS for j in POSITIONS]


def pair_file(system: System, i: int, j: int) -> str:
    s = system.sequence
    return f"pair_{i}_{s[i - 1]}__{j}_{s[j - 1]}.csv"


def pair_label(i: int, j: int, system: System | None = None) -> str:
    """'L1–I2' for a system, or 'L1–X6' generic when system is None."""
    seq = system.sequence if system else "LIVAGX"
    return f"{seq[i - 1]}{i}–{seq[j - 1]}{j}"


# Three-residue windows: start positions 1..4 -> LIV, IVA, VAG, AGX
WINDOW_STARTS = [1, 2, 3, 4]
WINDOW_PAIRS = [(a, b) for a in WINDOW_STARTS for b in WINDOW_STARTS if b >= a]  # 10


def window_name(system: System | None, start: int) -> str:
    seq = system.sequence if system else "LIVAGX"
    return seq[start - 1:start + 2]


def window_file(system: System, a: int, b: int) -> str:
    return f"{window_name(system, a)}__{window_name(system, b)}_halves.csv"


def window_label(a: int, b: int, system: System | None = None) -> str:
    return f"{window_name(system, a)}–{window_name(system, b)}"


# ------------------------------------------------------------------ loaders
@dataclass
class Curve:
    """A PMF curve on a common grid. Arrays share length; NaN where empty."""
    r: np.ndarray  # distance, nm
    pmf: np.ndarray  # kcal/mol (mean of halves, or full trajectory)
    err: np.ndarray | None = None  # |PMF1-PMF2|/2 where available
    half1: tuple[np.ndarray, np.ndarray] | None = None
    half2: tuple[np.ndarray, np.ndarray] | None = None


def _pmf_root(system: System, prefer_snapshot: bool = True) -> Path:
    if prefer_snapshot and system.snapshot_pmf.exists():
        return system.snapshot_pmf
    return system.pmf_dir


NORMALIZATION_DISTANCE = 1.8  # nm, as in pmf_from_cpptraj.py
PAIR_BIN = 0.01  # nm, single-residue pair PMFs
WINDOW_BIN = 0.02  # nm, three-residue window and LIVA fragment PMFs


def combine_halves(half1, half2, bin_width: float):
    """Same algorithm as combine_pmfs() in pmf_from_cpptraj.py: drop empty bins,
    shared grid from the larger start to min(smaller end, 1.8 nm), linear interp,
    mean and |PMF1-PMF2|/2."""
    valid = []
    for r, p in (half1, half2):
        ok = np.isfinite(r) & np.isfinite(p)
        if ok.sum() < 2:
            return np.array([]), np.array([]), np.array([])
        valid.append((r[ok], p[ok]))
    lo = max(r[0] for r, _ in valid)
    hi = min(min(r[-1] for r, _ in valid), NORMALIZATION_DISTANCE)
    if hi <= lo:
        return np.array([]), np.array([]), np.array([])
    grid = np.arange(lo, hi, bin_width)
    if grid.size == 0:
        grid = np.array([(lo + hi) / 2])
    a, b = (np.interp(grid, r, p) for r, p in valid)
    return grid, 0.5 * (a + b), 0.5 * np.abs(a - b)


def _read_halves(path: Path, bin_width: float) -> Curve:
    df = pd.read_csv(path)
    r1, p1, r2, p2 = (df.iloc[:, k].to_numpy(float) for k in range(4))
    m1, m2 = np.isfinite(r1), np.isfinite(r2)
    h1, h2 = (r1[m1], p1[m1]), (r2[m2], p2[m2])
    grid, mean, err = combine_halves(h1, h2, bin_width)
    return Curve(grid, mean, err, h1, h2)


def load_pair(system: System, i: int, j: int, kind: str = "mean") -> Curve:
    """kind = 'mean' (halves mean + |Δ|/2 band, as written by pmf_from_cpptraj)
    or 'full' (whole-trajectory PMF)."""
    root = _pmf_root(system)
    fname = pair_file(system, i, j)
    if kind == "full":
        df = pd.read_csv(root / "single_residue_pmf_full_value" / fname)
        r, p = df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)
        m = np.isfinite(r)
        return Curve(r[m], p[m])
    df = pd.read_csv(root / "single_residue_pmf_mean_value" / fname)
    r = df["common_distance_nm"].to_numpy(float)
    m = np.isfinite(r)
    c = _read_halves(root / "single_residue_pmf_halves_value" / fname, PAIR_BIN)
    return Curve(r[m], df["mean_pmf_kcal_per_mol"].to_numpy(float)[m],
                 df["abs_pmf_difference_div_2_kcal_per_mol"].to_numpy(float)[m],
                 c.half1, c.half2)


def load_window(system: System, a: int, b: int) -> Curve:
    root = _pmf_root(system)
    return _read_halves(root / "three_residue_pmf_halves_value" / window_file(system, a, b), WINDOW_BIN)


def load_fragment(system: System) -> Curve:
    root = _pmf_root(system)
    return _read_halves(root / "fragment_pmf_halves_value" / "LIVA_LIVA_halves.csv", WINDOW_BIN)


def well_depth(c: Curve) -> tuple[float, float]:
    """(min PMF in kcal/mol, distance of the minimum in nm). Reference is 0 at 1.8 nm,
    so a negative depth is an attractive well."""
    if not np.isfinite(c.pmf).any():
        return float("nan"), float("nan")
    k = int(np.nanargmin(c.pmf))
    return float(c.pmf[k]), float(c.r[k])


# ------------------------------------------------------------------ halved
def halved_dir(system: System, prefer_snapshot: bool = True) -> Path:
    snap = system.snapshot / "halved"
    return snap if prefer_snapshot and snap.exists() else system.analysis_dir / "halved"


def load_cpptraj_dat(path: Path) -> pd.DataFrame:
    """Read a cpptraj .dat (header line starting with '#Frame')."""
    with open(path) as fh:
        header = fh.readline().lstrip("#").split()
    return pd.read_csv(path, sep=r"\s+", comment="#", names=header, engine="c")


def load_hbonds(path: Path) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 7:
            continue
        rows.append(dict(acceptor=p[0], donor_h=p[1], donor=p[2], frames=int(p[3]),
                         frac=float(p[4]), avg_dist=float(p[5]), avg_ang=float(p[6])))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def register_figure(file: Path, title: str, caption: str, group: str,
                    systems: list[str] | None = None, order: int = 0) -> None:
    """Record a figure in figures/index.json so build_docs.py can place it.

    group: 'overview' | 'cross_system' | 'halved' | 'qc' | other free text.
    Safe to call repeatedly: the entry for the same file is replaced.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    rel = str(Path(file).resolve().relative_to(OVERALL))
    # several figure scripts may run at once: serialise index updates
    with open(FIGURES / ".index.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        idx = json.loads(FIGURE_INDEX.read_text()) if FIGURE_INDEX.exists() else []
        idx = [e for e in idx if e["file"] != rel]
        idx.append(dict(file=rel, title=title, caption=caption, group=group,
                        systems=systems or [s.key for s in SYSTEMS], order=order))
        idx.sort(key=lambda e: (e["group"], e["order"], e["file"]))
        FIGURE_INDEX.write_text(json.dumps(idx, indent=1, ensure_ascii=False))


def assert_inside_overall(path: Path) -> Path:
    p = Path(path).resolve()
    if OVERALL not in p.parents and p != OVERALL:
        raise PermissionError(f"refusing to write outside overall/: {p}")
    return p
