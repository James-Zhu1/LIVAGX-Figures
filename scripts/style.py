"""One matplotlib look for every overall/ figure.

Colours follow the validated reference data-viz palette:
  * systems   -> common.PALETTE slots 1-8, fixed per system (never by rank)
  * magnitude -> one-hue blue ramp (SEQ_BLUE)
  * signed    -> blue (negative / attractive) <-> red (positive), gray midpoint
Chrome is recessive: hairline solid grid, muted axes, text in neutral ink.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from common import FIGURES, assert_inside_overall, register_figure  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT_GRAY = "#c3c2b7"  # de-emphasised curves ("the rest")

# dark-mode steps of the same eight hues, for HTML/JS views
PALETTE_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                "#d55181", "#008300", "#9085e9", "#e66767"]

SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
# diverging: blue (attractive, < 0) -> gray -> red (repulsive, > 0)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f4a6a5", "#e34948", "#a32525"])

LINE_W = 1.4  # main data lines
THIN_W = 0.8  # context / many-curve overlays

PMF_LABEL = "PMF (kcal/mol)"
R_LABEL = "distance (nm)"


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.labelsize": 9, "axes.titleweight": "regular",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "lines.linewidth": LINE_W, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8,
    })


def save(fig, relpath: str, title: str, caption: str, group: str,
         systems: list[str] | None = None, order: int = 0, dpi: int = 160) -> Path:
    """Save under overall/figures/<relpath> and register it for the docs."""
    out = assert_inside_overall(FIGURES / relpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    register_figure(out, title, caption, group, systems, order)
    return out
