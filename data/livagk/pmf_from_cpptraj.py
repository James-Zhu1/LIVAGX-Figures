#!/usr/bin/env python3
"""PMF analysis for the LIVAGX two-peptide system, driven by cpptraj distances.

X is the sixth residue and is given on the command line:

    python3 pmf_from_cpptraj.py R --write-cpptraj   # -> pmf_distances_LIVAGR.in
    cpptraj -i pmf_distances_LIVAGR.in              # -> pmf_distances_LIVAGR.dat
    python3 pmf_from_cpptraj.py R                   # -> LIVAGR_PMF/

X accepts one-letter or three-letter codes (R, ARG, HIE, ...). With no
argument the sequence is LIVAGH.

Figures produced
  1. PMF_<seq>_6x6_halves_comparison.png   6x6 grid, first half vs second half
  2. PMF_<seq>_6x6_mean_uncertainty.png    6x6 grid, mean line + |PMF1-PMF2|/2 band
  3. PMF_<seq>_6x6_full_trajectory.png     6x6 grid from the whole trajectory
  4. PMF_<seq>_LIVA_LIVA_halves.png        LIVA-LIVA fragment, both halves
  5. PMF_<seq>_three_residue_halves.png    three-residue windows, both halves

Conventions kept from the mdtraj scripts: pair PMFs use a 0.01 nm density
histogram, fragment/window PMFs use a 0.02 nm count histogram, empty bins stay
NaN, every curve is shifted to zero at the bin nearest 1.8 nm, and the halves
are values[trim:mid] and values[mid+trim:] with trim = 5% of all frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ----------------------------------------------------------------- constants
K_B = 0.0019872041  # kcal/mol/K
TEMPERATURE = 300  # K
ANGSTROM_TO_NM = 0.1  # cpptraj writes Angstrom
PAIR_BIN_WIDTH = 0.01  # nm, single-residue pairs
GROUP_BIN_WIDTH = 0.02  # nm, LIVA and three-residue windows
NORMALIZATION_DISTANCE = 1.8  # nm
TRIM_FRACTION = 0.05  # frames dropped at the start of each half
WINDOW_SIZE = 3  # residues per three-residue window
FRAGMENT_LENGTH = 4  # LIVA fragment
PAIR_XLIM = (0.3, NORMALIZATION_DISTANCE)
GROUP_XLIM = (0.2, NORMALIZATION_DISTANCE)
PAIR_YLIM = (-1.5, 1.5)  # set to None for autoscaling
GROUP_YLIM = (-1.5, 1.5)
HALF_COLORS = ("tab:blue", "tab:orange")
DPI = 600

SEQUENCE_PREFIX = "LIVAG"  # the fixed part; X is appended to it
DEFAULT_X = "H"
DEFAULT_P1_FIRST = 2  # cpptraj residue number of L in peptide 1 (ACE is 1)
DEFAULT_P2_FIRST = 10  # cpptraj residue number of L in peptide 2

ONE_LETTER_CODES = set("ACDEFGHIKLMNPQRSTVWY")
THREE_LETTER_CODES = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # common Amber protonation-state names
    "HIE": "H", "HID": "H", "HIP": "H", "CYX": "C", "CYM": "C",
    "ASH": "D", "GLH": "E", "LYN": "K",
}


def amino_acid(text: str) -> str:
    """Turn 'R', 'arg' or 'HIE' into a single-letter code."""
    token = text.strip().upper()
    if len(token) == 1 and token in ONE_LETTER_CODES:
        return token
    if token in THREE_LETTER_CODES:
        return THREE_LETTER_CODES[token]
    raise argparse.ArgumentTypeError(
        f"'{text}' is not a recognized amino acid; use a one-letter code "
        "(R) or a three-letter code (ARG, HIE)."
    )


# --------------------------------------------------- cpptraj input generation
def build_cpptraj_input(
    sequence: str,
    topology: str,
    trajectory: str,
    distance_file: str,
    p1_first: int,
    p2_first: int,
) -> str:
    """Return the text of a cpptraj input file for this sequence."""
    length = len(sequence)
    p1_last = p1_first + length - 1
    p2_last = p2_first + length - 1
    windows = [
        sequence[start:start + WINDOW_SIZE]
        for start in range(length - WINDOW_SIZE + 1)
    ]
    fragment = sequence[:FRAGMENT_LENGTH]

    lines = []
    add = lines.append

    add("# ============================================================")
    add(f"#  PMF distance extraction for the two-peptide {sequence} system")
    add("#")
    add(f"#  run with:   cpptraj -i pmf_distances_{sequence}.in")
    add("#")
    add(f"#  Writes {distance_file}, which pmf_from_cpptraj.py reads.")
    add("#  All distances are center-of-mass distances (cpptraj default)")
    add("#  with the minimum-image convention applied automatically because")
    add("#  the trajectory carries box information.  cpptraj writes")
    add("#  ANGSTROM; the python script converts to nm.")
    add("#")
    add("#  Residue numbering (cpptraj is 1-based and counts the caps, so it")
    add("#  is the mdtraj 0-based resid + 1):")
    add(f"#    {p1_first - 1:>2}      ACE            cap, peptide 1")
    add(f"#    {p1_first:>2}-{p1_last:<4} {' '.join(sequence)}    peptide 1")
    add(f"#    {p1_last + 1:>2}      NME            cap, peptide 1")
    add(f"#    {p2_first - 1:>2}      ACE            cap, peptide 2")
    add(f"#    {p2_first:>2}-{p2_last:<4} {' '.join(sequence)}    peptide 2")
    add(f"#    {p2_last + 1:>2}      NME            cap, peptide 2")
    add("#  'resinfo' below prints the real numbering when you run this; if")
    add("#  it differs, re-generate with --p1-first / --p2-first.")
    add("#")
    add("#  Dataset names (the python script keys off these, not off X):")
    add("#    sr_i_j   residue i of peptide 1 vs residue j of peptide 2,")
    add(f"#             i,j = 1..{length}  ->  {' '.join(sequence)}")
    add(f"#    liva     {fragment} fragment of peptide 1 vs peptide 2")
    add("#    tr_i_j   three-residue window i of peptide 1 vs window j of")
    add(f"#             peptide 2, i,j = 1..{len(windows)}  ->  {' '.join(windows)}")
    add("# ============================================================")
    add("")
    add(f"parm {topology}")
    add(f"trajin {trajectory}")
    add("")
    add("# sanity check: prints the residue list, verify the masks below")
    add(f"resinfo :1-{p2_last + 1}")
    add("")

    add(f"# ---------- {length * length} single-residue pairs ----------")
    for i in range(length):
        add(
            f"#   {sequence[i]}{i + 1} (peptide 1, residue {p1_first + i}) "
            "vs peptide 2"
        )
        for j in range(length):
            name = f"sr_{i + 1}_{j + 1}"
            mask_a = f":{p1_first + i}"
            mask_b = f":{p2_first + j}"
            add(f"distance {name:<9} {mask_a:<8} {mask_b:<8} out {distance_file}")
        add("")

    add(f"# ---------- {fragment} - {fragment} fragment ----------")
    mask_a = f":{p1_first}-{p1_first + FRAGMENT_LENGTH - 1}"
    mask_b = f":{p2_first}-{p2_first + FRAGMENT_LENGTH - 1}"
    add(f"distance {'liva':<9} {mask_a:<8} {mask_b:<8} out {distance_file}")
    add("")

    add(f"# ---------- three-residue windows ({' '.join(windows)}) ----------")
    add("# all ordered combinations; the python script symmetrizes i <-> j")
    for i in range(len(windows)):
        for j in range(len(windows)):
            name = f"tr_{i + 1}_{j + 1}"
            mask_a = f":{p1_first + i}-{p1_first + i + WINDOW_SIZE - 1}"
            mask_b = f":{p2_first + j}-{p2_first + j + WINDOW_SIZE - 1}"
            comment = f"{windows[i]} - {windows[j]}"
            add(
                f"distance {name:<9} {mask_a:<8} {mask_b:<8} "
                f"out {distance_file}   # {comment}"
            )
        add("")

    add("# keep 5 decimals (Angstrom) so 0.1 A histogram bins stay clean")
    add(f"precision {distance_file} 12 5")
    add("")
    add("run")
    add("quit")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ file I/O
def read_cpptraj_distances(path: Path) -> dict[str, np.ndarray]:
    """Return {dataset name: distances in nm} from a cpptraj column file."""
    with open(path) as handle:
        header = handle.readline()
    if not header.startswith("#"):
        raise ValueError(f"{path} does not start with a cpptraj '#' header line.")

    names = header.lstrip("#").split()
    values = np.loadtxt(path, comments="#", ndmin=2)
    if values.shape[1] != len(names):
        raise ValueError(
            f"{path}: header has {len(names)} names but the data have "
            f"{values.shape[1]} columns."
        )

    return {
        name: values[:, column] * ANGSTROM_TO_NM
        for column, name in enumerate(names)
        if name.lower() != "frame"
    }


def require(data: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in data:
        raise KeyError(
            f"dataset '{name}' is missing from the cpptraj file; re-run the "
            "cpptraj input or re-generate it with --write-cpptraj."
        )
    return data[name]


# --------------------------------------------------------------- PMF helpers
def trimmed_halves(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split frame-aligned values in half and trim 5% from the start of each."""
    midpoint = len(values) // 2
    if midpoint == 0 or midpoint == len(values):
        raise ValueError("The trajectory must contain at least two frames.")

    length = int(len(values) * TRIM_FRACTION)
    return values[length:midpoint], values[midpoint + length:]


def trimmed_whole(values: np.ndarray) -> np.ndarray:
    """Drop the same leading 5% from the full trajectory (equilibration)."""
    length = int(len(values) * TRIM_FRACTION)
    if length >= len(values):
        raise ValueError("The trajectory is too short to trim.")
    return values[length:]


def normalized_pmf(
    distances: np.ndarray, bin_width: float, density: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return bin centers (nm) and PMF (kcal/mol) zeroed near 1.8 nm."""
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        raise ValueError("The distance array contains no finite values.")

    bins = np.arange(np.min(finite), np.max(finite) + bin_width, bin_width)
    histogram, bin_edges = np.histogram(finite, bins=bins, density=density)
    histogram = np.asarray(histogram, dtype=float)
    if not density:
        total = np.sum(histogram)
        if total == 0:
            raise ValueError("The histogram has zero total count.")
        histogram = histogram / total

    histogram = np.where(histogram == 0, np.nan, histogram)
    beta = 1 / (K_B * TEMPERATURE)
    pmf = -np.log(histogram) / beta
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    populated = np.isfinite(pmf)
    if not np.any(populated):
        raise ValueError("The histogram contains no populated bins.")

    candidates = np.abs(bin_centers - NORMALIZATION_DISTANCE)
    candidates[~populated] = np.inf
    normalized = pmf - pmf[np.argmin(candidates)]

    keep = bin_centers < NORMALIZATION_DISTANCE
    return bin_centers[keep], normalized[keep]


def half_pmfs_from(
    distances: np.ndarray, bin_width: float, density: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        normalized_pmf(half, bin_width, density)
        for half in trimmed_halves(distances)
    ]


def combine_pmfs(
    half_pmfs: list[tuple[np.ndarray, np.ndarray]], bin_width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Put both half-PMFs on a shared grid; return distance, mean, half spread."""
    valid_half_pmfs = []
    for distance, pmf in half_pmfs:
        valid = np.isfinite(distance) & np.isfinite(pmf)
        if np.count_nonzero(valid) < 2:
            raise ValueError("Each half-PMF needs at least two finite points.")
        valid_half_pmfs.append((distance[valid], pmf[valid]))

    common_min = max(distance[0] for distance, _ in valid_half_pmfs)
    common_max = min(
        min(distance[-1] for distance, _ in valid_half_pmfs),
        NORMALIZATION_DISTANCE,
    )
    if common_max <= common_min:
        raise ValueError("The two PMFs have no overlapping distance range.")

    common_distance = np.arange(common_min, common_max, bin_width)
    if common_distance.size == 0:
        common_distance = np.array([(common_min + common_max) / 2])

    interpolated = [
        np.interp(common_distance, distance, pmf)
        for distance, pmf in valid_half_pmfs
    ]
    mean_pmf = np.mean(interpolated, axis=0)
    half_difference = np.abs(interpolated[0] - interpolated[1]) / 2
    return common_distance, mean_pmf, half_difference


# --------------------------------------------------------------- CSV writing
def _padded(columns: list[np.ndarray]) -> np.ndarray:
    rows = max(len(column) for column in columns)
    values = np.full((rows, len(columns)), np.nan)
    for index, column in enumerate(columns):
        values[: len(column), index] = column
    return values


def save_curve_csv(path: Path, distance: np.ndarray, pmf: np.ndarray) -> None:
    np.savetxt(
        path,
        np.column_stack((distance, pmf)),
        delimiter=",",
        header="distance_nm,normalized_pmf_kcal_per_mol",
        comments="",
        fmt="%.10g",
    )


def save_halves_csv(
    path: Path, half_pmfs: list[tuple[np.ndarray, np.ndarray]]
) -> None:
    columns = []
    for distance, pmf in half_pmfs:
        columns.extend([distance, pmf])
    np.savetxt(
        path,
        _padded(columns),
        delimiter=",",
        header=(
            "distance_half_1_nm,normalized_pmf_half_1_kcal_per_mol,"
            "distance_half_2_nm,normalized_pmf_half_2_kcal_per_mol"
        ),
        comments="",
        fmt="%.10g",
    )


def save_mean_csv(
    path: Path,
    half_pmfs: list[tuple[np.ndarray, np.ndarray]],
    combined: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    common_distance, mean_pmf, half_difference = combined
    columns = []
    for distance, pmf in half_pmfs:
        columns.extend([distance, pmf])
    columns.extend([common_distance, mean_pmf, half_difference])
    np.savetxt(
        path,
        _padded(columns),
        delimiter=",",
        header=(
            "distance_half_1_nm,normalized_pmf_half_1_kcal_per_mol,"
            "distance_half_2_nm,normalized_pmf_half_2_kcal_per_mol,"
            "common_distance_nm,mean_pmf_kcal_per_mol,"
            "abs_pmf_difference_div_2_kcal_per_mol"
        ),
        comments="",
        fmt="%.10g",
    )


# ------------------------------------------------------------------ plotting
def style_axis(ax, title: str, xlim, ylim) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True)


def label_grid(axes) -> None:
    for ax in axes[-1, :]:
        ax.set_xlabel("Distance (nm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("PMF (kcal/mol)")


def pair_filename(sequence: str, i: int, j: int) -> str:
    return f"pair_{i}_{sequence[i - 1]}__{j}_{sequence[j - 1]}.csv"


def grid_figsize(sequence: str) -> tuple[float, float]:
    side = 3.34 * len(sequence)  # 20 x 20 inches for a six-residue peptide
    return side, side


# -------------------------------------------------------------- the analyses
def run_grid_halves(data, sequence, plot_dir, value_dir):
    """Figure 1: both trimmed halves on every panel."""
    n = len(sequence)
    fig, axes = plt.subplots(
        n, n, figsize=grid_figsize(sequence), sharex=True, sharey=True
    )
    fig.suptitle(
        f"{sequence}: PMFs from Two 5%-Trimmed Simulation Halves",
        fontsize=24,
        fontweight="bold",
    )

    for row in range(n):
        for column in range(n):
            i, j = row + 1, column + 1
            half_pmfs = half_pmfs_from(
                require(data, f"sr_{i}_{j}"), PAIR_BIN_WIDTH, True
            )
            save_halves_csv(value_dir / pair_filename(sequence, i, j), half_pmfs)

            ax = axes[row, column]
            for number, (distance, pmf) in enumerate(half_pmfs):
                ax.plot(
                    distance,
                    pmf,
                    label=f"Half {number + 1}",
                    color=HALF_COLORS[number],
                )
            style_axis(
                ax, f"{sequence[row]} - {sequence[column]}", PAIR_XLIM, PAIR_YLIM
            )

    axes[0, 0].legend()
    label_grid(axes)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(
        plot_dir / f"PMF_{sequence}_{n}x{n}_halves_comparison.png", dpi=DPI
    )
    plt.close(fig)


def run_grid_mean(data, sequence, plot_dir, value_dir):
    """Figure 2: mean of the two halves with a |PMF1-PMF2|/2 band."""
    n = len(sequence)
    fig, axes = plt.subplots(
        n, n, figsize=grid_figsize(sequence), sharex=True, sharey=True
    )
    fig.suptitle(
        f"{sequence}: Mean PMF with Half-Difference Uncertainty Band",
        fontsize=24,
        fontweight="bold",
    )

    for row in range(n):
        for column in range(n):
            i, j = row + 1, column + 1
            half_pmfs = half_pmfs_from(
                require(data, f"sr_{i}_{j}"), PAIR_BIN_WIDTH, True
            )
            combined = combine_pmfs(half_pmfs, PAIR_BIN_WIDTH)
            save_mean_csv(
                value_dir / pair_filename(sequence, i, j), half_pmfs, combined
            )

            common_distance, mean_pmf, half_difference = combined
            ax = axes[row, column]
            ax.plot(common_distance, mean_pmf, label="Mean PMF", color="tab:blue")
            ax.fill_between(
                common_distance,
                mean_pmf - half_difference,
                mean_pmf + half_difference,
                label="abs(PMF1 - PMF2) / 2",
                color="tab:blue",
                alpha=0.25,
            )
            style_axis(
                ax, f"{sequence[row]} - {sequence[column]}", PAIR_XLIM, PAIR_YLIM
            )

    axes[0, 0].legend()
    label_grid(axes)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(
        plot_dir / f"PMF_{sequence}_{n}x{n}_mean_uncertainty.png", dpi=DPI
    )
    plt.close(fig)


def run_grid_full(data, sequence, plot_dir, value_dir):
    """Figure 3: one PMF per panel from the whole (5%-trimmed) trajectory."""
    n = len(sequence)
    fig, axes = plt.subplots(
        n, n, figsize=grid_figsize(sequence), sharex=True, sharey=True
    )
    fig.suptitle(
        f"{sequence}: PMF from the Full Trajectory",
        fontsize=24,
        fontweight="bold",
    )

    for row in range(n):
        for column in range(n):
            i, j = row + 1, column + 1
            distances = trimmed_whole(require(data, f"sr_{i}_{j}"))
            distance, pmf = normalized_pmf(distances, PAIR_BIN_WIDTH, True)
            save_curve_csv(
                value_dir / pair_filename(sequence, i, j), distance, pmf
            )

            ax = axes[row, column]
            ax.plot(distance, pmf, label="Full trajectory", color="tab:blue")
            style_axis(
                ax, f"{sequence[row]} - {sequence[column]}", PAIR_XLIM, PAIR_YLIM
            )

    axes[0, 0].legend()
    label_grid(axes)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(
        plot_dir / f"PMF_{sequence}_{n}x{n}_full_trajectory.png", dpi=DPI
    )
    plt.close(fig)


def run_fragment_halves(data, sequence, plot_dir, value_dir):
    """Figure 4: LIVA-LIVA fragment PMF, first half vs second half."""
    fragment = sequence[:FRAGMENT_LENGTH]
    half_pmfs = half_pmfs_from(require(data, "liva"), GROUP_BIN_WIDTH, False)
    save_halves_csv(value_dir / f"{fragment}_{fragment}_halves.csv", half_pmfs)

    fig, ax = plt.subplots(figsize=(6, 6))
    for number, (distance, pmf) in enumerate(half_pmfs):
        ax.plot(
            distance, pmf, label=f"Half {number + 1}", color=HALF_COLORS[number]
        )
    style_axis(ax, f"{fragment} - {fragment} ({sequence})", GROUP_XLIM, GROUP_YLIM)
    ax.set_xticks(np.arange(0.2, 1.9, 0.4))
    ax.set_yticks(np.arange(-1, 1.51, 0.5))
    ax.set_xlabel("Distance (nm)", fontsize=12)
    ax.set_ylabel("PMF (kcal/mol)", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        plot_dir / f"PMF_{sequence}_{fragment}_{fragment}_halves.png", dpi=DPI
    )
    plt.close(fig)


def run_three_residue_halves(data, sequence, plot_dir, value_dir):
    """Figure 5: three-residue window pairs, first half vs second half.

    Off-diagonal panels pool the two equivalent orderings (window i of peptide 1
    with window j of peptide 2, and window j of peptide 1 with window i of
    peptide 2) inside each half, as the mdtraj notebook does.
    """
    windows = [
        sequence[start:start + WINDOW_SIZE]
        for start in range(len(sequence) - WINDOW_SIZE + 1)
    ]
    count = len(windows)

    fig, axes = plt.subplots(
        count, count, figsize=(count * 3.2, count * 3.2), squeeze=False
    )
    fig.suptitle(
        f"{sequence}: Three-Residue PMFs from Two Trimmed Halves",
        fontsize=22,
        fontweight="bold",
    )

    for i in range(count):
        for j in range(count):
            ax = axes[i, j]
            if i > j:
                ax.set_visible(False)
                continue

            halves = list(trimmed_halves(require(data, f"tr_{i + 1}_{j + 1}")))
            if i != j:
                swapped = trimmed_halves(require(data, f"tr_{j + 1}_{i + 1}"))
                halves = [
                    np.concatenate([halves[number], swapped[number]])
                    for number in (0, 1)
                ]

            half_pmfs = [
                normalized_pmf(half, GROUP_BIN_WIDTH, False) for half in halves
            ]
            save_halves_csv(
                value_dir / f"{windows[i]}__{windows[j]}_halves.csv", half_pmfs
            )

            for number, (distance, pmf) in enumerate(half_pmfs):
                ax.plot(
                    distance,
                    pmf,
                    label=f"Half {number + 1}",
                    color=HALF_COLORS[number],
                )
            style_axis(ax, f"{windows[i]} - {windows[j]}", GROUP_XLIM, GROUP_YLIM)
            ax.set_xticks(np.arange(0.2, 1.9, 0.4))
            ax.set_yticks(np.arange(-1, 1.51, 0.5))
            ax.set_xlabel("Distance (nm)", fontsize=9)
            ax.set_ylabel("PMF (kcal/mol)", fontsize=9)
            ax.tick_params(axis="both", labelsize=8)

    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(plot_dir / f"PMF_{sequence}_three_residue_halves.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PMF analysis for LIVAGX; X is given on the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "x",
        nargs="?",
        default=DEFAULT_X,
        type=amino_acid,
        help="last amino acid of the sequence, one- or three-letter code",
    )
    parser.add_argument(
        "--prefix",
        default=SEQUENCE_PREFIX,
        help="fixed part of the sequence that X is appended to",
    )
    parser.add_argument(
        "--write-cpptraj",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="write the cpptraj input for this sequence and exit "
        "(default path: pmf_distances_<SEQ>.in)",
    )
    parser.add_argument(
        "--distances",
        type=Path,
        default=None,
        help="cpptraj distance file (default: pmf_distances_<SEQ>.dat, "
        "falling back to pmf_distances.dat)",
    )
    parser.add_argument(
        "--topology",
        default=None,
        help="prmtop written into the cpptraj input "
        "(default: <sequence>_nowater.prmtop)",
    )
    parser.add_argument(
        "--trajectory",
        default="com_frames_nowater.dcd",
        help="trajectory written into the cpptraj input",
    )
    parser.add_argument(
        "--p1-first",
        type=int,
        default=DEFAULT_P1_FIRST,
        help="cpptraj residue number of the first residue of peptide 1",
    )
    parser.add_argument(
        "--p2-first",
        type=int,
        default=DEFAULT_P2_FIRST,
        help="cpptraj residue number of the first residue of peptide 2",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="output directory (default: <SEQ>_PMF next to the distance file)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    sequence = (arguments.prefix + arguments.x).upper()
    topology = arguments.topology or f"{sequence.lower()}_nowater.prmtop"
    distance_name = f"pmf_distances_{sequence}.dat"
    script = Path(__file__).name

    if arguments.write_cpptraj is not None:
        input_path = Path(
            arguments.write_cpptraj or f"pmf_distances_{sequence}.in"
        )
        input_path.write_text(
            build_cpptraj_input(
                sequence,
                topology,
                arguments.trajectory,
                distance_name,
                arguments.p1_first,
                arguments.p2_first,
            )
        )
        print(f"Wrote {input_path}  (parm {topology})")
        print(f"Next:  cpptraj -i {input_path}   ->  {distance_name}")
        print(f"Then:  python3 {script} {arguments.x}")
        return

    distance_path = arguments.distances
    if distance_path is None:
        distance_path = Path(distance_name)
        if not distance_path.exists():
            distance_path = Path("pmf_distances.dat")
    if not distance_path.exists():
        raise SystemExit(
            f"Distance file not found: {distance_path}\n"
            f"Generate the cpptraj input first:  "
            f"python3 {script} {arguments.x} --write-cpptraj"
        )

    data = read_cpptraj_distances(distance_path)

    output_root = arguments.outdir or (
        distance_path.resolve().parent / f"{sequence}_PMF"
    )
    plot_dir = output_root / "plots"
    value_dirs = {
        name: output_root / name
        for name in (
            "single_residue_pmf_halves_value",
            "single_residue_pmf_mean_value",
            "single_residue_pmf_full_value",
            "fragment_pmf_halves_value",
            "three_residue_pmf_halves_value",
        )
    }
    for directory in (plot_dir, *value_dirs.values()):
        directory.mkdir(parents=True, exist_ok=True)

    frames = len(next(iter(data.values())))
    print(f"Sequence: {sequence}   distances: {distance_path}")
    print(f"Datasets read: {len(data)}   frames: {frames}")

    run_grid_halves(
        data, sequence, plot_dir, value_dirs["single_residue_pmf_halves_value"]
    )
    print("1/5  grid, halves comparison")
    run_grid_mean(
        data, sequence, plot_dir, value_dirs["single_residue_pmf_mean_value"]
    )
    print("2/5  grid, mean with uncertainty band")
    run_grid_full(
        data, sequence, plot_dir, value_dirs["single_residue_pmf_full_value"]
    )
    print("3/5  grid, full trajectory")
    run_fragment_halves(
        data, sequence, plot_dir, value_dirs["fragment_pmf_halves_value"]
    )
    print(f"4/5  {sequence[:FRAGMENT_LENGTH]} fragment, halves")
    run_three_residue_halves(
        data, sequence, plot_dir, value_dirs["three_residue_pmf_halves_value"]
    )
    print("5/5  three-residue windows, halves")
    print(f"Done. Plots and CSVs under: {output_root}")


if __name__ == "__main__":
    main()
