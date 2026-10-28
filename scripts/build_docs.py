#!/usr/bin/env python3
"""Build the static documentation site for the LIVAGX PMF results.

    cd overall && python scripts/build_docs.py

Reads (never writes) the inputs below and writes HTML under overall/docs/:

  data/manifest.json, data/<key>/**   snapshot of each system's analysis folder
  figures/index.json, figures/**      new figures + CSV tables made by the other
                                      overall/scripts (may be partial or missing)
  git log of overall/                 for the changelog (read-only git commands)

Output
  docs/index.html, figures.html, data.html, notes.html, changelog.html
  docs/systems/<key>.html             one page per system
  docs/compare/index.html, <type>.html  one page per original plot type, 8 systems
  docs/assets/site.css, site.js       shared look + lightbox / filters / theme
  docs/systems/previews/**            downscaled WebP copies of the very large
                                      original PNGs (up to 12024 x 12024 px), made
                                      only if Pillow is importable; without it the
                                      pages fall back to the original PNGs.

The site works from file:// (relative links only, no fetch, no CDN).
docs/explorer.html, docs/assets/explorer.* and docs/assets/pmf_data.js belong to
another script and are never written here.
"""

from __future__ import annotations

import html
import json
import re
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BY_KEY, DATA, DOCS, FIGURE_INDEX, FIGURES, MANIFEST,  # noqa: E402
                    NORMALIZATION_DISTANCE, OVERALL, PAIR_BIN, POSITIONS, PROJECT,
                    SYSTEMS, WINDOW_BIN, WINDOW_PAIRS, WINDOW_STARTS, System,
                    assert_inside_overall, halved_dir, load_cpptraj_dat,
                    load_fragment, load_hbonds, load_pair, load_window, pair_label,
                    well_depth, window_name)

# Colours: light palette from common.SYSTEMS, dark steps + diverging map from
# style.py when importable (it pulls in matplotlib); otherwise identical copies.
DIV_LIGHT = ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f4a6a5", "#e34948", "#a32525"]
try:
    import style as _style  # noqa: E402
    PALETTE_DARK = list(_style.PALETTE_DARK)
    _CMAP_DIV = _style.CMAP_DIV
except Exception:  # pragma: no cover - fallback keeps the generator stdlib+pandas
    PALETTE_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                    "#d55181", "#008300", "#9085e9", "#e66767"]
    _CMAP_DIV = None
# dark-mode diverging steps: magnitude -> brighter against the dark surface
DIV_DARK = ["#9ec5f4", "#3987e5", "#1f4f8c", "#2c2c2a", "#8c2f2e", "#e66767", "#f4a6a5"]
SEQ_LIGHT = ["#cde2fb", "#2a78d6"]

try:
    from PIL import Image  # Pillow ships with matplotlib; only used for previews
    Image.MAX_IMAGE_PIXELS = None  # the source 6x6 grids are 12024 x 12024 px
except Exception:  # pragma: no cover
    Image = None

SITE = "LIVAGX PMFs"
THEME_KEY = "livagx-docs-theme"  # same key as explorer.html, so the choice carries over
PREVIEWS = DOCS / "systems" / "previews"
THUMB_W, VIEW_W = 1000, 2400
FRAMES_PER_HALF = 270_000
TAIL_NM = 1.6  # a minimum at r >= 1.6 nm is the flat tail, not a contact well (as in fig_overview_grid)
PROTECTED = {"explorer.html", "pmf_data.js"}

GROUPS = [("overview", "Overview"), ("cross_system", "Across systems"),
          ("halved", "Halved structure"), ("qc", "Convergence QC")]
GROUP_BLURB = {
    "overview": "Every PMF in one figure: the 6×6 residue-pair grid, the three-residue windows and LIVA–LIVA, all eight systems overlaid in their fixed colours.",
    "cross_system": "The same PMF overlaid for all eight systems, well-depth maps and residue-6 (X) focus.",
    "halved": "Structure of LIVA-LIVA contact frames per trajectory half: contacts, H-bonds, angle, end-to-end distance, backbone dihedrals.",
    "qc": "Convergence checks: first half vs second half, full trajectory vs mean of halves.",
}

# Original plot types, in display order. Keys are the file names with the
# sequence-specific part stripped (see plot_type()).
PLOT_TYPES = {
    "6x6_full_trajectory": (
        "Pair PMFs, full trajectory (6×6)", "pmf",
        "36 residue-pair PMFs from the whole (5%-trimmed) trajectory: row = residue on peptide 1, "
        "column = residue on peptide 2, 0.01 nm bins, zero at 1.8 nm. The last row and column involve "
        "X6, the residue that differs between systems. Compare depth and position of the first contact "
        "minimum (about 0.5–0.7 nm)."),
    "6x6_halves_comparison": (
        "Pair PMFs, first vs second half (6×6)", "pmf",
        "Same 36 pair PMFs computed separately from the first and the second half of the trajectory "
        "(5% trimmed at the start of each half). Where the two curves overlap the PMF is converged; "
        "a gap is sampling noise."),
    "6x6_mean_uncertainty": (
        "Pair PMFs, mean of halves with |PMF₁−PMF₂|/2 band (6×6)", "pmf",
        "Mean of the two half-trajectory PMFs on a common grid; the shaded band is half the absolute "
        "difference between the halves. These mean curves are what the well-depth tables use."),
    "liva_liva_halves": (
        "LIVA–LIVA fragment PMF, both halves", "pmf",
        "PMF of the distance between the LIVA fragments of the two peptides (residues 2–5 vs 10–13), "
        "first vs second half, 0.02 nm bins, zero at 1.8 nm."),
    "three_residue_halves": (
        "Three-residue window PMFs, both halves", "pmf",
        "PMFs between three-residue windows (LIV, IVA, VAG, AGX) of the two peptides, first vs second "
        "half, 0.02 nm bins. Off-diagonal panels pool both orderings (window i on peptide 1 with j on "
        "peptide 2 and vice versa)."),
    "angle_histogram_first_vs_second_half": (
        "Peptide 1 – peptide 2 angle histogram, first vs second half", "halved",
        "Distribution of the angle between the two peptide direction vectors in LIVA-LIVA contact "
        "frames (fragment distance < 7 Å), first vs second half."),
    "end_to_end_dist_histogram_first_vs_second_half": (
        "End-to-end distance histogram (peptide 1), first vs second half", "halved",
        "Distribution of the peptide-1 end-to-end distance (Cα of L1 to Cα of X6, "
        ":2@CA–:7@CA) over all frames of each half."),
    "hbond_heatmaps_first_second_half": (
        "Inter-peptide H-bond heatmaps, first vs second half", "halved",
        "Fraction of LIVA–LIVA contact frames with each donor → acceptor inter-peptide hydrogen bond, "
        "per residue, for residues L, I, V, A of both peptides (2–5, 10–13)."),
    "hbond_network_first_second_half": (
        "Inter-peptide H-bond network, first vs second half", "halved",
        "The same hydrogen bonds drawn as a directed network between the LIVA residues of the two "
        "peptides; edge width grows with the occupancy fraction."),
}
ANGLE_CAVEAT = ("The source cpptraj input defines v1 = :2 → :5 (Leu → Ala of peptide 1) but "
                "v2 = :9 → :12 (ACE → Val of peptide 2); the matching vector would be :10 → :13, so "
                "this angle is likely off by one residue on peptide 2.")
LABEL_CAVEAT = ("Plot titles and file names in the three LIVAGE folders say LIVAGH_CC / _CN / _NN: "
                "a label typo in combined_first_second_half.py. The data are the LIVAGE system.")
CAVEATS_BY_TYPE = {
    "angle_histogram_first_vs_second_half": [ANGLE_CAVEAT, LABEL_CAVEAT],
    "end_to_end_dist_histogram_first_vs_second_half": [LABEL_CAVEAT],
    "hbond_heatmaps_first_second_half": [LABEL_CAVEAT],
    "hbond_network_first_second_half": [LABEL_CAVEAT],
}
# keyword hints linking an original plot type to newly registered figures
RELATED = {
    "6x6_full_trajectory": ["pairs_6x6_full", "full_vs_mean", "big_grid_full"],
    "6x6_halves_comparison": ["halves", "half1_vs_half2", "rmsd", "convergence"],
    "6x6_mean_uncertainty": ["pairs_6x6_mean", "well_depth", "well_position", "big_grid_mean"],
    "liva_liva_halves": ["fragment", "liva"],
    "three_residue_halves": ["window"],
    "angle_histogram_first_vs_second_half": ["angle"],
    "end_to_end_dist_histogram_first_vs_second_half": ["end_to_end", "end-to-end"],
    "hbond_heatmaps_first_second_half": ["hbond"],
    "hbond_network_first_second_half": ["hbond"],
}
# meaningful subsets for the compare pages
PRESETS = [
    ("All", [s.key for s in SYSTEMS]),
    ("LIVAGE protonation states", ["livage_cc", "livage_cn", "livage_nn"]),
    ("Cationic X (H⁺, K, R)", ["livagh", "livagk", "livagr"]),
    ("Neutral X (G, Q, E⁰/E⁰)", ["livagg", "livagq", "livage_nn"]),
]

NOW = datetime.now().astimezone()
WRITTEN: list[Path] = []


# ================================================================== helpers
def esc(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def href(depth: int, path: str) -> str:
    """URL from a page `depth` folders below docs/ to `path` (relative to docs/)."""
    frag = ""
    if "#" in path:
        path, frag = path.split("#", 1)
        frag = "#" + quote(frag)
    return esc("../" * depth + quote(path) + frag)


def pth(path) -> str:
    """Escaped path that may line-break only after a slash."""
    return esc(path).replace("/", "/<wbr>")


def data_path(key: str, rel: str) -> str:
    """docs-root-relative path of a snapshot file."""
    return f"../data/{key}/{rel}"


def overall_path(rel: str) -> str:
    return f"../{rel}"


def human_size(n) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def num(x, nd=2, unit="") -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "–"
    if not np.isfinite(x):
        return "–"
    s = f"{x:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return f"{s.replace('-', '−')}{unit}"


def png_size(path: Path) -> tuple[int, int] | None:
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", head[16:24])
    except OSError:
        pass
    return None


def write(path: Path, text: str) -> None:
    p = assert_inside_overall(path)
    if p.name in PROTECTED or p.name.startswith("explorer."):
        raise PermissionError(f"{p} belongs to the explorer script")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    WRITTEN.append(p)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def sw(key: str) -> str:
    return f'<span class="sw" style="--c:var(--sys-{esc(key)})" aria-hidden="true"></span>'


def sys_chip(s: System, depth: int, link: bool = True) -> str:
    inner = f"{sw(s.key)}{esc(s.label)}"
    if link:
        return f'<a class="syslink" href="{href(depth, f"systems/{s.key}.html")}">{inner}</a>'
    return f'<span class="syslink">{inner}</span>'


# ------------------------------------------------------------------ colours
def _rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(max(0.0, min(1.0, c)) * 255)) for c in rgb[:3])


def _interp(anchors: list[str], t: float) -> str:
    t = max(0.0, min(1.0, t))
    n = len(anchors) - 1
    x = t * n
    i = min(int(x), n - 1)
    f = x - i
    a, b = _rgb(anchors[i]), _rgb(anchors[i + 1])
    return _hex([a[k] + (b[k] - a[k]) * f for k in range(3)])


def _lum(h: str) -> float:
    def ch(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(c) for c in _rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def div_colors(v: float, vmax: float) -> tuple[str, str, str, str]:
    """(light bg, light text, dark bg, dark text) for a signed value."""
    t = 0.5 + 0.5 * v / vmax
    light = _hex(_CMAP_DIV(max(0.0, min(1.0, t)))) if _CMAP_DIV is not None else _interp(DIV_LIGHT, t)
    dark = _interp(DIV_DARK, t)
    tl = "#0b0b0b" if _lum(light) > 0.32 else "#f9f9f7"
    td = "#0b0b0b" if _lum(dark) > 0.32 else "#f0efec"
    return light, tl, dark, td


def nice_max(values, floor=0.5, step=0.25) -> float:
    v = [abs(x) for x in values if x is not None and np.isfinite(x)]
    m = max(v) if v else floor
    return max(floor, float(np.ceil(m / step) * step))


# ================================================================== inputs
def load_manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: cannot read {MANIFEST}: {exc}")
        return {}


def load_figure_index() -> list[dict]:
    if not FIGURE_INDEX.exists():
        return []
    try:
        idx = json.loads(FIGURE_INDEX.read_text())
    except Exception as exc:  # noqa: BLE001  (another script may be mid-write)
        print(f"WARNING: figures/index.json unreadable ({exc}); building without new figures")
        return []
    out = []
    for e in idx:
        f = OVERALL / e.get("file", "")
        if not e.get("file") or not f.exists():
            print(f"WARNING: registered figure missing on disk: {e.get('file')}")
            continue
        e = dict(e)
        e.setdefault("title", Path(e["file"]).stem)
        e.setdefault("caption", "")
        e.setdefault("group", "other")
        e.setdefault("systems", [s.key for s in SYSTEMS])
        e.setdefault("order", 0)
        e["id"] = "fig-" + slug(Path(e["file"]).with_suffix("").as_posix())
        out.append(e)
    out.sort(key=lambda e: (e["order"], e["file"]))
    return out


def plot_type(stem: str) -> str:
    """'PMF_LIVAGH_6x6_full_trajectory' -> '6x6_full_trajectory';
    'hbond_heatmaps_LIVAGH_CC_tripled_time_new_ff_first_second_half' ->
    'hbond_heatmaps_first_second_half'."""
    s = re.sub(r"^PMF_", "", stem)
    s = re.sub(r"LIVAG[A-Z](?:_(?:CC|CN|NN))?(?![A-Z])", "", s)
    s = re.sub(r"tripled_time_new_ff(?:_doubled)?", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


def plot_title(ptype: str) -> str:
    if ptype in PLOT_TYPES:
        return PLOT_TYPES[ptype][0]
    return ptype.replace("_", " ").capitalize()


@dataclass
class Img:
    """One image as shown on a page: thumbnail, view-size and original."""
    src: Path  # original file
    thumb: str  # docs-root-relative
    view: str
    full: str
    w: int | None = None
    h: int | None = None
    tw: int | None = None
    th: int | None = None
    size: int = 0
    vw: int | None = None
    vh: int | None = None


@dataclass
class Plot:
    key: str
    rel: str  # path inside data/<key>/
    ptype: str
    title: str
    section: str  # 'pmf' | 'halved' | 'other'
    img: Img


# ------------------------------------------------------------------ previews
class Previewer:
    """Downscaled WebP copies of huge PNGs, cached by source fingerprint."""

    def __init__(self):
        self.cache_file = PREVIEWS / ".cache.json"
        try:
            self.cache = json.loads(self.cache_file.read_text())
        except Exception:  # noqa: BLE001
            self.cache = {}
        self.used: set[Path] = set()
        self.made = 0

    def _render(self, src: Path, outs: list[tuple[Path, int]]) -> list[tuple[int, int]]:
        im = Image.open(src)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        dims = []
        for out, width in sorted(outs, key=lambda o: -o[1]):
            if im.width > width:
                factor = max(1, im.width // (width * 2))
                if factor > 1:
                    im = im.reduce(factor)
                h = round(im.height * width / im.width)
                im = im.resize((width, h), Image.LANCZOS)
            out.parent.mkdir(parents=True, exist_ok=True)
            im.save(assert_inside_overall(out), "WEBP", quality=88 if width >= 2000 else 84, method=4)
            dims.append((im.width, im.height))
        return dims[::-1]

    def get(self, src: Path, out_stem: Path, fingerprint: str) -> Img:
        """out_stem: docs/systems/previews/<...>/<name> (without extension)."""
        size = src.stat().st_size if src.exists() else 0
        full = "../" + src.resolve().relative_to(OVERALL).as_posix()
        dims = png_size(src)
        img = Img(src, full, full, full, *(dims or (None, None)), *(dims or (None, None)), size=size)
        img.vw, img.vh = img.w, img.h
        if Image is None or dims is None:
            return img
        w, h = dims
        if w * h < 6e6 and size < 2e6:  # small enough to show as is
            return img
        thumb, view = out_stem.with_suffix(".thumb.webp"), out_stem.with_suffix(".view.webp")
        rel_t = thumb.relative_to(DOCS).as_posix()
        rel_v = view.relative_to(DOCS).as_posix()
        entry = self.cache.get(rel_v)
        if not (entry and entry.get("fp") == fingerprint and thumb.exists() and view.exists()):
            try:
                (tw, th), (vw, vh) = self._render(src, [(thumb, THUMB_W), (view, VIEW_W)])
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: preview failed for {src}: {exc}")
                return img
            entry = dict(fp=fingerprint, thumb=[tw, th], view=[vw, vh])
            self.cache[rel_v] = entry
            self.made += 1
            print(f"  preview {rel_v} ({vw}x{vh})", flush=True)
        self.used.update({thumb.resolve(), view.resolve()})
        img.thumb, img.view = rel_t, rel_v
        img.tw, img.th = entry["thumb"]
        img.vw, img.vh = entry["view"]
        return img

    def finish(self) -> None:
        if not PREVIEWS.exists():
            return
        for f in PREVIEWS.rglob("*.webp"):
            if f.resolve() not in self.used:
                assert_inside_overall(f).unlink()
        live = {p.relative_to(DOCS.resolve()).as_posix() for p in self.used}
        self.cache = {k: v for k, v in self.cache.items() if k in live}
        PREVIEWS.mkdir(parents=True, exist_ok=True)
        write(self.cache_file, json.dumps(self.cache, indent=1))


# ------------------------------------------------------------------ per system data
@dataclass
class SysData:
    s: System
    man: dict
    plots: list[Plot] = field(default_factory=list)
    pair_mean: dict = field(default_factory=dict)  # (i,j) -> (depth, r, err)
    pair_full: dict = field(default_factory=dict)
    windows: dict = field(default_factory=dict)  # (a,b) -> (depth, r, err)
    fragment: tuple = (np.nan, np.nan, np.nan)
    halves: dict = field(default_factory=dict)  # 'first'/'second' -> stats dict
    hb: dict = field(default_factory=dict)  # 'first'/'second' -> DataFrame

    @property
    def files(self) -> list[dict]:
        return self.man.get("files", [])

    def counts(self) -> dict:
        c = {"copied": 0, "too_large": 0, "excluded": 0}
        for f in self.files:
            c[f.get("status", "?")] = c.get(f.get("status", "?"), 0) + 1
        return c

    def deepest_pair(self):
        best = None
        for (i, j), (d, r, e) in self.pair_mean.items():
            if np.isfinite(d) and (best is None or d < best[1][0]):
                best = ((i, j), (d, r, e))
        return best


def _depth(curve) -> tuple[float, float, float]:
    d, r = well_depth(curve)
    err = np.nan
    if curve.err is not None and np.isfinite(d):
        k = int(np.nanargmin(curve.pmf))
        err = float(curve.err[k])
    return d, r, err


def _safe(fn, *a):
    try:
        return _depth(fn(*a))
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: {fn.__name__}{a[1:]} for {a[0].key}: {exc}")
        return (np.nan, np.nan, np.nan)


def _summary_csv() -> pd.DataFrame | None:
    p = FIGURES / "halved" / "halved_summary.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return None


def halved_stats(s: System, summary: pd.DataFrame | None) -> dict:
    out = {}
    if summary is not None and {"system", "half", "contact_frames"}.issubset(summary.columns):
        for _, row in summary[summary["system"] == s.key].iterrows():
            out[str(row["half"])] = dict(
                contact=int(row["contact_frames"]),
                pct=float(row.get("contact_pct", 100 * row["contact_frames"] / FRAMES_PER_HALF)),
                hb=float(row.get("total_hbond_frac", np.nan)),
                e2e=float(row.get("e2e_mean_A", np.nan)),
                angle=float(row.get("angle_mean_deg", np.nan)))
        if len(out) == 2:
            return out
    hd = halved_dir(s)
    for half in ("first", "second"):
        st = {}
        try:
            a = load_cpptraj_dat(hd / f"pep1pep2_angle_{half}_half.dat")
            st.update(contact=len(a), pct=100 * len(a) / FRAMES_PER_HALF, angle=float(a.iloc[:, 1].mean()))
        except Exception:  # noqa: BLE001
            pass
        try:
            e = load_cpptraj_dat(hd / f"end_to_end_dist_{half}_half.dat")
            st["e2e"] = float(e.iloc[:, 1].mean())
        except Exception:  # noqa: BLE001
            pass
        try:
            st["hb"] = float(load_hbonds(hd / f"interhbond_{half}_half.txt")["frac"].sum())
        except Exception:  # noqa: BLE001
            pass
        out[half] = st
    return out


def collect(manifest: dict, prev: Previewer) -> list[SysData]:
    summary = _summary_csv()
    out = []
    for s in SYSTEMS:
        sd = SysData(s, manifest.get(s.key, {}))
        print(f"{s.key}: wells, H-bonds, images", flush=True)
        for i, j in [(i, j) for i in POSITIONS for j in POSITIONS]:
            sd.pair_mean[(i, j)] = _safe(load_pair, s, i, j, "mean")
            sd.pair_full[(i, j)] = _safe(load_pair, s, i, j, "full")
        for a, b in WINDOW_PAIRS:
            sd.windows[(a, b)] = _safe(load_window, s, a, b)
        sd.fragment = _safe(load_fragment, s)
        sd.halves = halved_stats(s, summary)
        for half in ("first", "second"):
            try:
                df = load_hbonds(halved_dir(s) / f"interhbond_{half}_half.txt")
                sd.hb[half] = df.sort_values("frac", ascending=False).reset_index(drop=True)
            except Exception:  # noqa: BLE001
                sd.hb[half] = pd.DataFrame()
        pngs = [f for f in sd.files if f.get("status") == "copied" and f["path"].lower().endswith(".png")]
        for f in pngs:
            src = DATA / s.key / f["path"]
            if not src.exists():
                continue
            stem = Path(f["path"]).stem
            pt = plot_type(stem)
            section = ("pmf" if "_PMF/plots/" in f"{f['path']}" or f["path"].startswith(f"{s.sequence}_PMF/")
                       else "halved" if f["path"].startswith("halved/") else "other")
            img = prev.get(src, PREVIEWS / s.key / stem, f.get("sha256") or f"{f['size']}-{f['mtime']}")
            sd.plots.append(Plot(s.key, f["path"], pt, plot_title(pt), section, img))
        order = {k: n for n, k in enumerate(PLOT_TYPES)}
        sd.plots.sort(key=lambda p: (order.get(p.ptype, 99), p.rel))
        out.append(sd)
    return out


def figure_images(figs: list[dict], prev: Previewer) -> dict[str, Img]:
    out = {}
    for e in figs:
        src = OVERALL / e["file"]
        st = src.stat()
        stem = PREVIEWS / "_figures" / Path(e["file"]).with_suffix("")
        out[e["file"]] = prev.get(src, stem, f"{st.st_size}-{int(st.st_mtime)}")
    return out


# ------------------------------------------------------------------ git
def git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(OVERALL), "-c", "core.quotepath=off", *args],
                           capture_output=True, text=True, timeout=120)
    except Exception:  # noqa: BLE001
        return None
    return r.stdout if r.returncode == 0 else None


def parse_git_log(text: str) -> list[dict]:
    """Parse `git log --name-status --format=%x1e%H%x1f%h%x1f%ad%x1f%an%x1f%D%x1f%s`."""
    commits = []
    for rec in text.split("\x1e"):
        if not rec.strip():
            continue
        lines = rec.split("\n")
        head = lines[0].split("\x1f")
        if len(head) < 6:
            continue
        full, short, date, author, refs, subject = head[:5] + ["\x1f".join(head[5:])]
        changes = []
        for ln in lines[1:]:
            if not ln.strip():
                continue
            parts = ln.split("\t")
            st = parts[0][:1]
            if st in "RC" and len(parts) >= 3:
                changes.append((st, parts[2], parts[1]))
            elif len(parts) >= 2:
                changes.append((st, parts[1], None))
        tags = [r.strip()[5:] for r in refs.split(",") if r.strip().startswith("tag: ")]
        commits.append(dict(hash=full, short=short, date=date, author=author, refs=refs,
                            tags=tags, subject=subject, changes=changes))
    return commits


def git_history() -> tuple[list[dict], str | None]:
    head = git("rev-parse", "--short", "HEAD")
    text = git("log", "--name-status", "--date=iso-strict",
               "--format=%x1e%H%x1f%h%x1f%ad%x1f%an%x1f%D%x1f%s")
    return (parse_git_log(text) if text else []), (head.strip() if head else None)


def change_group(path: str) -> str:
    parts = path.split("/")
    if parts[0] == "data" and len(parts) > 2 and parts[1] in BY_KEY:
        return parts[1]
    if path == "data/manifest.json":
        return "manifest"
    if parts[0] in ("figures", "docs", "scripts"):
        return parts[0]
    return "other"


# ================================================================== page shell
NAV = [("index.html", "Home"), ("figures.html", "Figures"), ("compare/index.html", "Compare"),
       ("explorer.html", "Explorer"), ("systems", "Systems"), ("data.html", "Data"),
       ("notes.html", "Notes"), ("changelog.html", "Changelog")]


def nav(depth: int, active: str) -> str:
    items = []
    for target, label in NAV:
        if target == "systems":
            links = "".join(
                f'<a href="{href(depth, f"systems/{s.key}.html")}"'
                f'{" aria-current=page" if active == "sys:" + s.key else ""}>'
                f'{sw(s.key)}<span>{esc(s.label)}</span><span class="muted mono">{esc(s.x_resname)}</span></a>'
                for s in SYSTEMS)
            cur = ' class="cur"' if active.startswith("sys:") else ""
            items.append(f'<details class="dropdown"><summary{cur}>Systems</summary>'
                         f'<div class="menu">{links}</div></details>')
            continue
        cur = ' aria-current="page"' if active == target else ""
        items.append(f'<a class="nl" href="{href(depth, target)}"{cur}>{esc(label)}</a>')
    return (f'<header class="topbar"><nav class="nav" aria-label="Site">'
            f'<a class="brand" href="{href(depth, "index.html")}">LIVAGX <span>PMFs · 60 mM parallel</span></a>'
            f'{"".join(items)}<span class="spacer"></span>'
            f'<button class="theme-toggle" type="button" data-theme-toggle>Theme</button></nav></header>')


def page(title: str, body: str, depth: int, active: str, wide: bool = False, desc: str = "") -> str:
    up = "../" * depth
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} · {SITE}</title>
<meta name="description" content="{esc(desc or title)}">
<meta name="generator" content="overall/scripts/build_docs.py">
<link rel="stylesheet" href="{up}assets/site.css">
<script>try{{var t=localStorage.getItem('{THEME_KEY}');if(t==='light'||t==='dark')document.documentElement.setAttribute('data-theme',t);}}catch(e){{}}</script>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
{nav(depth, active)}
<main id="main" class="container{' wide' if wide else ''}">
{body}
</main>
<footer class="site-footer"><div class="inner">
Generated by <code>overall/scripts/build_docs.py</code> on {esc(NOW.strftime('%Y-%m-%d %H:%M %Z'))}.
Source analysis folders (read-only): <code>{esc(PROJECT)}</code>.
</div></footer>
<script src="{up}assets/site.js" defer></script>
</body>
</html>
"""


def seq_chips(s: System | None) -> str:
    seq = s.sequence if s else "LIVAGX"
    key = s.key if s else None

    def pep(n):
        cells = ['<span class="res cap">ACE</span>']
        for k, aa in enumerate(seq, 1):
            x = ' x' if k == 6 else ""
            style = f' style="--c:var(--sys-{key})"' if (k == 6 and key) else ""
            cells.append(f'<span class="res{x}"{style} title="peptide {n}, position {k}">{esc(aa)}<sub>{k}</sub></span>')
        cells.append('<span class="res cap">NHE</span>')
        return f'<span class="chain"><span class="pepl">P{n}</span>' + "".join(cells) + "</span>"

    return f'<div class="seq" aria-label="sequence">{pep(1)}{pep(2)}</div>'


def img_link(img: Img, depth: int, group: str, title: str, sub: str = "", color: str = "",
             alt: str = "", lazy: bool = True, use_thumb: bool = True) -> str:
    src = img.thumb if use_thumb else img.view
    w, h = (img.tw, img.th) if use_thumb else (img.vw, img.vh)
    dims = f' width="{w}" height="{h}"' if w and h else ""
    attrs = (f'class="shot lb" href="{href(depth, img.view)}" data-lb="{esc(group)}" '
             f'data-title="{esc(title)}" data-sub="{esc(sub)}" data-full="{href(depth, img.full)}"')
    if color:
        attrs += f' data-color="{esc(color)}"'
    loading = ' loading="lazy"' if lazy else ""
    return (f'<a {attrs}><img src="{href(depth, src)}" alt="{esc(alt or title)}"{dims}'
            f'{loading} decoding="async"></a>')


def full_note(img: Img, depth: int) -> str:
    dims = f", {img.w}×{img.h} px" if img.w else ""
    return f'<a href="{href(depth, img.full)}">original PNG</a> <span class="muted">({human_size(img.size)}{dims})</span>'


# ================================================================== tables
def heat_legend(vmax: float, label: str) -> str:
    light = ", ".join(_interp(DIV_LIGHT, k / 6) if _CMAP_DIV is None else _hex(_CMAP_DIV(k / 6))
                      for k in range(7))
    dark = ", ".join(DIV_DARK)
    return (f'<div class="legend"><span class="ink2">{esc(label)} (kcal/mol)</span>'
            f'<span class="legend-scale"><span class="legend-bar" style="--gl:linear-gradient(90deg,{light});'
            f'--gd:linear-gradient(90deg,{dark})"></span><span class="legend-ticks"><span>{num(-vmax, 2)}</span>'
            f'<span>0</span><span>{num(vmax, 2)}</span></span></span>'
            f'<span class="muted">blue = attractive well, below the 1.8 nm reference</span>'
            f'<span class="muted"><span class="tailchip">0.00</span> grey italic = minimum at r ≥ {TAIL_NM} nm, '
            f'i.e. no contact well</span></div>')


def heat_cell(val: tuple, vmax: float, tip: str, show_err: bool = True) -> str:
    d, r, e = val
    if not np.isfinite(d):
        return f'<td class="na" title="{esc(tip)}: no data">–</td>'
    err = f" ± {num(e, 2)}" if (show_err and np.isfinite(e)) else ""
    title = f"{tip}: {num(d, 2)}{err} kcal/mol at {num(r, 2)} nm"
    if r >= TAIL_NM:
        return (f'<td class="tail" title="{esc(title)} (minimum in the tail: no contact well)">'
                f'{num(d, 2)}<small>{num(r, 2)} nm</small></td>')
    bl, tl, bd, td = div_colors(d, vmax)
    return (f'<td class="v" style="--hl:{bl};--tl:{tl};--hd:{bd};--td:{td}" title="{esc(title)}">'
            f'{num(d, 2)}<small>{num(r, 2)} nm</small></td>')


def pair_table(sd: SysData, which: str, vmax: float) -> str:
    vals = sd.pair_mean if which == "mean" else sd.pair_full
    seq = sd.s.sequence
    head = "".join(f"<th scope=col>{esc(seq[j - 1])}{j}</th>" for j in POSITIONS)
    rows = []
    for i in POSITIONS:
        cells = "".join(heat_cell(vals[(i, j)], vmax, pair_label(i, j, sd.s), which == "mean")
                        for j in POSITIONS)
        rows.append(f"<tr><th scope=row>{esc(seq[i - 1])}{i}</th>{cells}</tr>")
    return (f'<div class="table-wrap plain"><table class="heat"><thead><tr>'
            f'<th class="corner"><span>pep 1 ↓</span><span>pep 2 →</span></th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def window_table(sd: SysData, vmax: float) -> str:
    names = [window_name(sd.s, a) for a in WINDOW_STARTS]
    head = "".join(f"<th scope=col>{esc(n)}</th>" for n in names)
    rows = []
    for a in WINDOW_STARTS:
        cells = []
        for b in WINDOW_STARTS:
            if b < a:
                cells.append('<td class="blank" aria-hidden="true"></td>')
            else:
                cells.append(heat_cell(sd.windows[(a, b)], vmax,
                                       f"{window_name(sd.s, a)}–{window_name(sd.s, b)}"))
        rows.append(f"<tr><th scope=row>{esc(names[a - 1])}</th>{''.join(cells)}</tr>")
    return (f'<div class="table-wrap plain"><table class="heat"><thead><tr><th class="corner"></th>{head}'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


BACKBONE = {"N", "H", "O", "C", "CA", "HA", "HN1", "HN2", "CH3", "OXT"}


def hb_atom(token: str, s: System) -> str:
    m = re.match(r"([A-Z0-9]+)_(\d+)@(\S+)", token)
    if not m:
        return esc(token)
    num_, atom = int(m.group(2)), m.group(3)
    pep, pos = (1, num_ - 1) if num_ <= 8 else (2, num_ - 9)
    name = "ACE" if pos == 0 else "NHE" if pos == 7 else f"{s.sequence[pos - 1]}{pos}"
    return (f'<span class="pep" title="{esc(token)}">P{pep}</span> {esc(name)} '
            f'<code>{esc(atom)}</code>')


def hb_table(df: pd.DataFrame, s: System, contact: int | None) -> str:
    if df is None or df.empty:
        return '<p class="muted">No H-bond file in the snapshot.</p>'
    top = df.head(10)
    vmax = max(float(df["frac"].max()), 1e-9)
    rows = []
    for k, r in enumerate(top.itertuples(), 1):
        kind = ("BB" if r.donor.split("@")[-1] in BACKBONE else "SC") + "→" + \
               ("BB" if r.acceptor.split("@")[-1] in BACKBONE else "SC")
        p = 100 * r.frac / vmax
        rows.append(
            f"<tr><td class=num>{k}</td><td>{hb_atom(r.donor, s)}</td><td>{hb_atom(r.acceptor, s)}</td>"
            f"<td class=muted>{kind}</td>"
            f'<td class="num bar" style="--p:{p:.1f}%">{100 * r.frac:.1f}%</td>'
            f"<td class=num>{r.frames:,}</td><td class=num>{r.avg_dist:.2f}</td><td class=num>{r.avg_ang:.1f}</td></tr>")
    total = f"{df['frac'].sum():.2f}"
    foot = (f'<p class="small muted">{len(df)} distinct H-bonds in this half; sum of fractions = {total} '
            f'(≈ mean number of inter-peptide H-bonds per contact frame)'
            f'{f"; contact frames = {contact:,}" if contact else ""}.</p>')
    return (f'<div class="table-wrap"><table class="tbl hb"><thead><tr><th>#</th><th>Donor</th><th>Acceptor</th>'
            f'<th>Type</th><th>Fraction</th><th>Frames</th><th>d (Å)</th><th>angle (°)</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>{foot}')


def status_badge(st: str) -> str:
    return f'<span class="badge {esc(st)}">{esc(st.replace("_", " "))}</span>'


def inventory_rows(sd: SysData, depth: int, with_system: bool = False) -> str:
    rows = []
    for f in sorted(sd.files, key=lambda f: f["path"]):
        st = f.get("status", "?")
        path = pth(f["path"])
        if st == "copied":
            cell = f'<a href="{href(depth, data_path(sd.s.key, f["path"]))}">{path}</a>'
        else:
            cell = f'{path}<div class="src">original: <code>{pth(PROJECT / f.get("source", ""))}</code></div>'
        sysc = f"<td>{sys_chip(sd.s, depth)}</td>" if with_system else ""
        rows.append(
            f'<tr data-status="{esc(st)}" data-system="{esc(sd.s.key)}">{sysc}<td class="path">{cell}</td>'
            f'<td class="num" data-v="{f.get("size", 0)}">{human_size(f.get("size", 0))}</td><td>{status_badge(st)}</td>'
            f'<td class="mono muted">{esc((f.get("sha256") or "")[:10]) or "–"}</td>'
            f'<td class="muted nowrap">{esc(f.get("mtime", "").replace("T", " "))}</td></tr>')
    return "".join(rows)


def filter_bar(table_id: str, systems: bool = False, unit: str = "files") -> str:
    sys_sel = ""
    if systems:
        opts = "".join(f'<option value="{esc(s.key)}">{esc(s.label)}</option>' for s in SYSTEMS)
        sys_sel = f'<select data-attr="system" aria-label="System"><option value="">All systems</option>{opts}</select>'
    return (f'<div class="filters" data-table="#{table_id}" data-unit="{unit}">'
            f'<input type="search" placeholder="Filter by name, folder, status…" aria-label="Filter rows">'
            f'{sys_sel}<select data-attr="status" aria-label="Status"><option value="">Any status</option>'
            f'<option value="copied">copied</option><option value="too_large">too large</option>'
            f'<option value="excluded">excluded</option></select><span class="count" data-count></span></div>')


# ================================================================== figure blocks
def csvs_for(fig_file: str, figs: list[dict]) -> list[Path]:
    p = OVERALL / fig_file
    csvs = sorted(p.parent.glob("*.csv"))
    if not csvs:
        return []
    stem = p.stem.lower()
    exact = [c for c in csvs if c.stem.lower().startswith(stem) or stem.startswith(c.stem.lower())]
    if exact:
        return exact
    siblings = [e for e in figs if (OVERALL / e["file"]).parent == p.parent]
    if len(siblings) <= 1:
        return csvs
    generic = {"vs", "and", "of", "the", "all", "per", "by", "x", "png", "csv"}

    def tok(t):
        return {w for w in re.split(r"[^a-z0-9]+", t.lower()) if w and w not in generic}

    tk = tok(stem)
    return [c for c in csvs if tok(c.stem) & (tk - {"mean", "full"})]


def systems_note(keys: list[str], depth: int) -> str:
    keys = [k for k in keys if k in BY_KEY]
    if not keys or set(keys) == set(BY_KEY):
        return '<span>all 8 systems</span>'
    return " ".join(f'<span class="nowrap">{sw(k)}{esc(BY_KEY[k].label)}</span>' for k in keys)


def figure_block(e: dict, img: Img, depth: int, figs: list[dict], wide: bool = False,
                 heading: str = "h3") -> str:
    csv_links = " · ".join(
        f'<a href="{href(depth, overall_path(c.relative_to(OVERALL).as_posix()))}">{esc(c.name)}</a>'
        for c in csvs_for(e["file"], figs))
    csv_html = f'<span>tables: {csv_links}</span>' if csv_links else ""
    link = img_link(img, depth, "figs-" + e["group"], e["title"], e.get("caption", "")[:160],
                    alt=e["title"], use_thumb=not wide)
    return (f'<figure class="fig{" wide" if wide else ""}" id="{esc(e["id"])}">{link}<figcaption>'
            f'<{heading}>{esc(e["title"])}</{heading}><p>{esc(e.get("caption", ""))}</p>'
            f'<div class="meta"><span>{systems_note(e.get("systems", []), depth)}</span>'
            f'<a href="{href(depth, overall_path(e["file"]))}"><code>{pth(e["file"])}</code></a>'
            f'<span class="muted">{human_size(img.size)}{f", {img.w}×{img.h} px" if img.w else ""}</span>'
            f'{csv_html}</div></figcaption></figure>')


# ================================================================== pages
def build_index(sds, figs, fimgs, commits, head, manifest) -> None:
    depth = 0
    by_group = {g: [e for e in figs if e["group"] == g] for g, _ in GROUPS}
    totals = {"copied": 0, "too_large": 0, "excluded": 0}
    for sd in sds:
        for k, v in sd.counts().items():
            totals[k] = totals.get(k, 0) + v
    synced = datetime.fromtimestamp(MANIFEST.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if MANIFEST.exists() else "–"
    tag = next((c["tags"][0] for c in commits if c["tags"]), None)
    head_html = (f'<code>{esc(head)}</code>{f" · latest tag <code>{esc(tag)}</code>" if tag else ""}'
                 if head else '<span class="muted">no snapshots committed yet</span>')

    # deepest well: prefer the overview CSV (same numbers the figures use), else computed
    well_csv = FIGURES / "overview" / "pair_well_depths_mean.csv"
    deepest = {}
    try:
        w = pd.read_csv(well_csv)
        for k, g in w.groupby("system"):
            r = g.loc[g["min_pmf_kcal"].idxmin()]
            deepest[k] = (float(r["min_pmf_kcal"]), float(r["r_min_nm"]), int(r["i"]), int(r["j"]))
    except Exception:  # noqa: BLE001
        pass

    rows = []
    for sd in sds:
        s = sd.s
        if s.key in deepest:
            d, r, i, j = deepest[s.key]
        else:
            best = sd.deepest_pair()
            (i, j), (d, r, _) = best if best else ((1, 1), (np.nan, np.nan, np.nan))
        h1, h2 = sd.halves.get("first", {}), sd.halves.get("second", {})
        contact = (f'{h1.get("contact", 0):,} / {h2.get("contact", 0):,}'
                   f'<div class="muted small">{num(h1.get("pct"), 1)}% / {num(h2.get("pct"), 1)}% of 270,000</div>'
                   if h1.get("contact") else "–")
        fd, fr, fe = sd.fragment
        rows.append(
            f"<tr><td>{sys_chip(s, depth)}</td><td class=mono>{esc(s.sequence)}</td>"
            f"<td class=mono>{esc(s.x_resname)}</td><td>{esc(s.note)}</td><td class=num>{contact}</td>"
            f"<td class=num>{num(d)} <span class=muted>kcal/mol</span>"
            f"<div class='muted small'>{esc(pair_label(i, j, s))} at {num(r)} nm</div></td>"
            f"<td class=num>{num(fd)} <span class=muted>± {num(fe)}</span>"
            f"<div class='muted small'>at {num(fr)} nm</div></td>"
            f'<td class="path small"><code title="{esc(s.analysis_dir)}">{pth(s.analysis_dir.relative_to(PROJECT))}</code></td></tr>')
    systems_table = (
        '<div class="table-wrap"><table class="tbl systems"><thead><tr><th>System</th><th>Sequence</th>'
        '<th>Residue 6 (Amber)</th><th>Charge / protonation</th><th>LIVA–LIVA contact frames<br>'
        '<span class="muted">half 1 / half 2</span></th><th>Deepest pair well<br><span class="muted">mean of halves</span></th>'
        '<th>LIVA–LIVA well<br><span class="muted">mean of halves</span></th><th>Source folder</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>')

    ov = by_group.get("overview", [])
    ov_html = "".join(figure_block(e, fimgs[e["file"]], depth, figs, wide=True) for e in ov) or \
        '<p class="note">Overview figures have not been generated yet. Run <code>./update.sh</code> in <code>overall/</code>.</p>'

    n_types = len({p.ptype for sd in sds for p in sd.plots})
    cards = [
        ("compare/index.html", "Compare one plot across systems",
         "Each original plot type side by side for all eight systems, with a lightbox to flip between them.",
         f"{n_types} plot types × {len(sds)} systems"),
        ("figures.html#cross_system", "Across systems", GROUP_BLURB["cross_system"],
         f"{len(by_group.get('cross_system', []))} figures"),
        ("figures.html#halved", "Halved structure", GROUP_BLURB["halved"],
         f"{len(by_group.get('halved', []))} figures"),
        ("figures.html#qc", "Convergence QC", GROUP_BLURB["qc"], f"{len(by_group.get('qc', []))} figures"),
        ("explorer.html", "Interactive explorer",
         "Pick systems and residue pairs and overlay their PMF curves in the browser.", "interactive"),
        ("data.html", "Data inventory",
         "Every file of every analysis folder: what was copied, what is too large, where originals live.",
         f"{sum(totals.values()):,} files"),
        ("notes.html", "Methods & notes",
         "PMF definition, halves, frame ranges, residue numbering, known caveats, how to update.", "read first"),
        ("changelog.html", "Changelog", "One entry per git snapshot, with the data files that changed per system.",
         f"{len(commits)} snapshots" if commits else "no snapshots yet"),
    ]
    cards_html = "".join(
        f'<a class="card" href="{href(depth, t)}"><h3>{esc(h)}</h3><p>{esc(p)}</p><span class="count">{esc(c)}</span></a>'
        for t, h, p, c in cards)

    body = f"""
<section class="hero">
  <p class="eyebrow">Two-peptide MD · 60 mM · parallel · Amber · 300 K</p>
  <h1>LIVAGX peptide PMFs — 60 mM parallel</h1>
  <p class="lede">Potentials of mean force between two LIVAG<b>X</b> hexapeptides for eight variants of residue 6
  (E in three protonation states, G, H<sup>+</sup>, K, Q, R): 36 residue pairs, ten three-residue window pairs and the
  LIVA fragment, each from two trajectory halves; plus the structure of LIVA–LIVA contact frames per half.</p>
  {seq_chips(None)}
  <dl class="facts">
    <div><dt>Snapshot built</dt><dd>{esc(NOW.strftime('%Y-%m-%d %H:%M %Z'))}</dd></div>
    <div><dt>Git HEAD at build</dt><dd>{head_html}</dd></div>
    <div><dt>Data synced</dt><dd>{esc(synced)}</dd></div>
    <div><dt>Snapshot files</dt><dd>{totals.get('copied', 0):,} copied · {totals.get('too_large', 0)} too large · {totals.get('excluded', 0)} excluded</dd></div>
  </dl>
</section>

<h2 id="systems">Systems</h2>
<p class="ink2">Click a system for its original plots, well-depth tables, top H-bonds and file inventory.
Contact frames are frames with LIVA–LIVA distance &lt; 7 Å out of 270,000 per half. Wells are the minimum of the
mean-of-halves PMF (zero at 1.8 nm); ± is |PMF<sub>1</sub>−PMF<sub>2</sub>|/2 at the minimum.</p>
{systems_table}

<h2 id="overview">Overview</h2>
<p class="ink2">{esc(GROUP_BLURB['overview'])} Click a figure to open it full size.</p>
{ov_html}

<h2 id="explore">Explore</h2>
<div class="cards">{cards_html}</div>
"""
    write(DOCS / "index.html", page("Home", body, depth, "index.html",
                                    desc="LIVAGX two-peptide PMF results, 60 mM parallel"))


def build_system_pages(sds, figs, fimgs, vmax_pair, vmax_win) -> None:
    depth = 1
    n = len(sds)
    for k, sd in enumerate(sds):
        s = sd.s
        prev_sd, next_sd = sds[(k - 1) % n], sds[(k + 1) % n]
        pager = (f'<nav class="pager" aria-label="System navigation">'
                 f'<a href="{href(depth, f"systems/{prev_sd.s.key}.html")}">← {sw(prev_sd.s.key)}{esc(prev_sd.s.label)}</a>'
                 f'<a href="{href(depth, f"systems/{next_sd.s.key}.html")}">{sw(next_sd.s.key)}{esc(next_sd.s.label)} →</a></nav>')
        c = sd.counts()
        h1, h2 = sd.halves.get("first", {}), sd.halves.get("second", {})
        best = sd.deepest_pair()
        xx = sd.pair_mean[(6, 6)]
        fd, fr, fe = sd.fragment

        def tile(label, value, detail=""):
            return f'<div class="stat"><div class="k">{label}</div><div class="v">{value}</div><div class="d">{detail}</div></div>'

        def well_tile(label, w):
            d, r, e = w
            if np.isfinite(r) and r >= TAIL_NM:
                return tile(label, '<span class="muted">no well</span>',
                            f'minimum {num(d)} kcal/mol at {num(r)} nm (flat tail)')
            return tile(label, f'{num(d)} <small>kcal/mol</small>', f'at {num(r)} nm · ± {num(e)}')

        tiles = [
            tile("LIVA–LIVA contact frames", f'{h1.get("contact", 0):,} / {h2.get("contact", 0):,}',
                 f'half 1 / half 2 · {num(h1.get("pct"), 1)}% / {num(h2.get("pct"), 1)}% of 270,000'),
            tile("Deepest pair well", f'{num(best[1][0])} <small>kcal/mol</small>' if best else "–",
                 f'{esc(pair_label(*best[0], s))} at {num(best[1][1])} nm · ± {num(best[1][2])}' if best else ""),
            well_tile("LIVA–LIVA fragment well", sd.fragment),
            well_tile(f"{esc(s.x)}6–{esc(s.x)}6 well", xx),
            tile("H-bonds per contact frame", f'{num(h1.get("hb"))} / {num(h2.get("hb"))}',
                 "sum of occupancy fractions, half 1 / half 2"),
            tile("Mean end-to-end distance", f'{num(h1.get("e2e"), 1)} / {num(h2.get("e2e"), 1)} <small>Å</small>',
                 "peptide 1, :2@CA–:7@CA, all frames"),
            tile("Mean pep1–pep2 angle", f'{num(h1.get("angle"), 1)} / {num(h2.get("angle"), 1)} <small>°</small>',
                 'contact frames · <a href="../notes.html#caveats">see caveat</a>'),
        ]

        def plot_cards(section):
            out = []
            for p in [p for p in sd.plots if p.section == section]:
                notes = CAVEATS_BY_TYPE.get(p.ptype, [])
                notes = [x for x in notes if x != LABEL_CAVEAT or s.sequence == "LIVAGE"]
                note_html = "".join(f'<p class="small muted">Caveat: {esc(x)}</p>' for x in notes)
                out.append(
                    f'<figure class="fig">{img_link(p.img, depth, "sys", p.title, s.label, f"var(--sys-{s.key})")}'
                    f'<figcaption><h3>{esc(p.title)}</h3>'
                    f'<p class="small"><a href="{href(depth, f"compare/{p.ptype}.html")}">Compare across systems →</a></p>'
                    f'{note_html}<div class="meta"><code>{pth(p.rel)}</code>{full_note(p.img, depth)}</div>'
                    f'</figcaption></figure>')
            return "".join(out) or '<p class="muted">No plots of this kind in the snapshot.</p>'

        own_figs = [e for e in figs if e.get("systems") == [s.key]]
        own_html = ""
        if own_figs:
            own_html = ('<h2 id="new-figures">New figures for this system</h2><div class="fig-grid">' +
                        "".join(figure_block(e, fimgs[e["file"]], depth, figs) for e in own_figs) + "</div>")

        pmf_root = f"{s.sequence}_PMF"
        folders = [
            (f"{pmf_root}/single_residue_pmf_mean_value/", "Pair PMFs, mean of halves + |Δ|/2 (36 CSV)"),
            (f"{pmf_root}/single_residue_pmf_halves_value/", "Pair PMFs, half 1 and half 2 (36 CSV)"),
            (f"{pmf_root}/single_residue_pmf_full_value/", "Pair PMFs, full trajectory (36 CSV)"),
            (f"{pmf_root}/three_residue_pmf_halves_value/", "Three-residue window PMFs, halves (10 CSV)"),
            (f"{pmf_root}/fragment_pmf_halves_value/", "LIVA–LIVA fragment PMF, halves (1 CSV)"),
            (f"{pmf_root}/plots/", "Original PMF plots (5 PNG)"),
            ("halved/", "Halved analysis: H-bond tables, angle and end-to-end .dat, plots, script"),
            ("pmf_from_cpptraj.py", "PMF script used for this system"),
        ]
        folder_rows = []
        for rel, desc in folders:
            exists = (DATA / s.key / rel.rstrip("/")).exists()
            nfiles = sum(1 for f in sd.files if f.get("status") == "copied" and f["path"].startswith(rel)) \
                if rel.endswith("/") else None
            link = (f'<a href="{href(depth, data_path(s.key, rel))}"><code>{esc(rel)}</code></a>' if exists
                    else f'<code>{esc(rel)}</code> <span class="muted">(missing)</span>')
            folder_rows.append(f"<tr><td class=path>{link}</td><td>{esc(desc)}</td>"
                               f"<td class=num>{'' if nfiles is None else nfiles}</td></tr>")

        label_note = ""
        if s.sequence == "LIVAGE":
            label_note = f'<p class="note warn"><b>Label caveat.</b> {esc(LABEL_CAVEAT)}</p>'
        if s.key == "livagr":
            label_note = ('<p class="note warn"><b>Folder name.</b> The source folder is spelled '
                          '<code>analysis_contined</code>; the snapshot stores it as <code>data/livagr/</code> like the others.</p>')
        if s.key == "livagh":
            label_note = ('<p class="note warn"><b>Script label.</b> <code>halved/combined_first_second_half.py</code> now '
                          'sets the label <code>LIVAGH_tripled_time_new_ff_doubled</code>, but the halved PNGs were written '
                          'before that edit and are named without <code>_doubled</code>.</p>')

        body = f"""
<p class="crumb"><a href="../index.html#systems">Systems</a> / {esc(s.label)}</p>
<div class="sys-head" style="--c:var(--sys-{esc(s.key)})">
  <h1>{sw(s.key)}{esc(s.label)}</h1>
  {pager}
</div>
{seq_chips(s)}
<dl class="facts">
  <div><dt>Residue 6 (Amber)</dt><dd class="mono">{esc(s.x_resname)}</dd></div>
  <div><dt>Charge / protonation</dt><dd>{esc(s.note)}</dd></div>
  <div><dt>Source folder (read-only)</dt><dd><code>{pth(s.analysis_dir)}</code></dd></div>
  <div><dt>Snapshot</dt><dd><a href="{href(depth, f"../data/{s.key}/")}"><code>data/{esc(s.key)}/</code></a> · {c.get('copied', 0)} copied · {c.get('too_large', 0)} too large · {c.get('excluded', 0)} excluded</dd></div>
</dl>
{label_note}

<h2 id="glance">At a glance</h2>
<div class="stats">{"".join(tiles)}</div>

<h2 id="pmf-plots">Original PMF plots</h2>
<p class="ink2">From <code>{esc(pmf_root)}/plots/</code>, made by <code>pmf_from_cpptraj.py</code>. Click to enlarge; ←/→ flips through this system's plots.</p>
<div class="fig-grid">{plot_cards("pmf")}</div>

<h2 id="halved-plots">Original halved-analysis plots</h2>
<p class="ink2">From <code>halved/</code>, made by <code>combined_first_second_half.py</code> on LIVA–LIVA contact frames (distance &lt; 7 Å) of each half.</p>
<div class="fig-grid">{plot_cards("halved")}</div>

<h2 id="wells">Pair well depths</h2>
<p class="ink2">Minimum of each residue-pair PMF (r &lt; 1.8 nm, zero at 1.8 nm) and its distance. Rows = residue on peptide 1,
columns = residue on peptide 2. Same colour scale on every system page, so pages can be compared. Hover a cell for the value and,
for the mean, ± |PMF<sub>1</sub>−PMF<sub>2</sub>|/2 at the minimum.</p>
{heat_legend(vmax_pair, "well depth")}
<div class="two">
  <div><h3>Mean of halves</h3>{pair_table(sd, "mean", vmax_pair)}</div>
  <div><h3>Full trajectory</h3>{pair_table(sd, "full", vmax_pair)}</div>
</div>

<h2 id="windows">Three-residue windows and LIVA fragment</h2>
<p class="ink2">Mean-of-halves well depth for the ten unordered window pairs (off-diagonal cells pool both orderings).
LIVA–LIVA fragment: <b>{num(fd)} ± {num(fe)} kcal/mol</b> at {num(fr)} nm.</p>
{heat_legend(vmax_win, "well depth")}
{window_table(sd, vmax_win)}

<h2 id="hbonds">Top inter-peptide H-bonds</h2>
<p class="ink2">From <code>halved/interhbond_*_half.txt</code> (cpptraj <code>hbond :2-5,10-13 nointramol</code>, residues L, I, V, A only).
Fraction = share of LIVA–LIVA contact frames in which the bond is present. P1 / P2 = peptide; BB = backbone atom, SC = side chain.</p>
<div class="two">
  <div><h3>First half</h3>{hb_table(sd.hb.get("first"), s, h1.get("contact"))}</div>
  <div><h3>Second half</h3>{hb_table(sd.hb.get("second"), s, h2.get("contact"))}</div>
</div>
{own_html}
<h2 id="raw">Raw data folders</h2>
<div class="table-wrap"><table class="tbl"><thead><tr><th>Folder / file</th><th>Contents</th><th>Files</th></tr></thead>
<tbody>{"".join(folder_rows)}</tbody></table></div>

<h2 id="inventory">Snapshot inventory</h2>
<p class="ink2">Every file in the source analysis folder. <b>copied</b> files are in the snapshot (sha256 prefix shown);
<b>too large</b> (&gt; 20 MB) and <b>excluded</b> (<code>*.in *.prmtop *.log *.nc</code>) files stay only in the source folder.</p>
{filter_bar("inv-" + s.key)}
<div class="table-wrap tall"><table class="tbl" id="inv-{esc(s.key)}"><thead><tr><th>Path</th><th>Size</th><th>Status</th><th>sha256</th><th>Source modified</th></tr></thead>
<tbody>{inventory_rows(sd, depth)}</tbody></table></div>
{pager}
"""
        write(DOCS / "systems" / f"{s.key}.html",
              page(s.label, body, depth, "sys:" + s.key, desc=f"{s.label}: original plots, wells, H-bonds, files"))


def related_figs(ptype: str, figs: list[dict]) -> list[dict]:
    keys = RELATED.get(ptype, [])
    out = []
    for e in figs:
        hay = (e["file"] + " " + e["title"]).lower()
        if any(k in hay for k in keys):
            out.append(e)
    return out


def build_compare(sds, figs) -> None:
    depth = 1
    types = [t for t in PLOT_TYPES if any(p.ptype == t for sd in sds for p in sd.plots)]
    types += sorted({p.ptype for sd in sds for p in sd.plots} - set(types))
    by = {t: {sd.s.key: next((p for p in sd.plots if p.ptype == t), None) for sd in sds} for t in types}

    cards = []
    for t in types:
        title, kind, desc = PLOT_TYPES.get(t, (plot_title(t), "other", ""))
        first = next((p for p in by[t].values() if p), None)
        thumb = (f'<img src="{href(depth, first.img.thumb)}" alt="" loading="lazy" decoding="async"'
                 f'{f" width={first.img.tw} height={first.img.th}" if first and first.img.tw else ""}>' if first else "")
        n = sum(1 for p in by[t].values() if p)
        cards.append(f'<a class="card pt" href="{href(depth, f"compare/{t}.html")}"><span class="pt-thumb">{thumb}</span>'
                     f'<h3>{esc(title)}</h3><p>{esc(desc[:170] + ("…" if len(desc) > 170 else ""))}</p>'
                     f'<span class="count">{"PMF" if kind == "pmf" else "halved analysis"} · {n} systems</span></a>')
    body = f"""
<h1>Compare one plot across systems</h1>
<p class="lede">The original plots of each analysis folder, one page per plot type, all eight systems side by side.
Open any image to flip through the systems with ← / →. For overlays on shared axes see <a href="../figures.html#cross_system">Across systems</a>
and the <a href="../explorer.html">interactive explorer</a>.</p>
<div class="cards">{"".join(cards)}</div>
"""
    write(DOCS / "compare" / "index.html", page("Compare", body, depth, "compare/index.html",
                                                desc="Same original plot for all eight systems"))

    for k, t in enumerate(types):
        title, kind, desc = PLOT_TYPES.get(t, (plot_title(t), "other", ""))
        grid = []
        for sd in sds:
            p = by[t][sd.s.key]
            s = sd.s
            if p is None:
                grid.append(f'<article class="cmp-card" data-key="{esc(s.key)}" style="--c:var(--sys-{esc(s.key)})">'
                            f'<header>{sw(s.key)}<b>{esc(s.label)}</b></header><p class="muted pad">Not present in this system\'s folder.</p></article>')
                continue
            grid.append(
                f'<article class="cmp-card" data-key="{esc(s.key)}" style="--c:var(--sys-{esc(s.key)})">'
                f'<header>{sw(s.key)}<b>{esc(s.label)}</b><span class="muted mono">{esc(s.sequence)} · {esc(s.x_resname)}</span></header>'
                f'{img_link(p.img, depth, "cmp", s.label, title, f"var(--sys-{s.key})", alt=f"{s.label}: {title}")}'
                f'<footer><a href="{href(depth, f"systems/{s.key}.html")}">System page</a> · {full_note(p.img, depth)}'
                f'<div class="mono small muted path">{pth(p.rel)}</div></footer></article>')
        chips = "".join(f'<button type="button" class="chip" data-key="{esc(s.key)}" aria-pressed="true" '
                        f'style="--c:var(--sys-{esc(s.key)})">{sw(s.key)}{esc(s.label)}</button>' for s in SYSTEMS)
        presets = "".join(f'<button type="button" data-preset="{esc(",".join(keys))}">{esc(lbl)}</button>'
                          for lbl, keys in PRESETS)
        cav = "".join(f'<p class="note warn"><b>Caveat.</b> {esc(x)}</p>' for x in CAVEATS_BY_TYPE.get(t, []))
        rel = related_figs(t, figs)
        rel_html = ""
        if rel:
            rel_html = ('<h2 id="related">Related new figures</h2><ul class="links">' + "".join(
                f'<li><a href="{href(depth, "figures.html#" + e["id"])}">{esc(e["title"])}</a> '
                f'<span class="muted">· {esc(dict(GROUPS).get(e["group"], e["group"]))}</span></li>' for e in rel) + "</ul>")
        prev_t, next_t = types[(k - 1) % len(types)], types[(k + 1) % len(types)]
        pager = (f'<nav class="pager" aria-label="Plot type navigation">'
                 f'<a href="{href(depth, f"compare/{prev_t}.html")}">← {esc(plot_title(prev_t))}</a>'
                 f'<a href="{href(depth, f"compare/{next_t}.html")}">{esc(plot_title(next_t))} →</a></nav>')
        body = f"""
<p class="crumb"><a href="index.html">Compare</a> / {esc(title)}</p>
<h1>{esc(title)}</h1>
<p class="lede">{esc(desc)}</p>
{cav}
<div class="controls">
  <div class="ctl"><span class="ctl-label">Columns</span><div class="seg" data-cols-for="#grid">
    <button type="button" data-cols="auto">Auto</button><button type="button" data-cols="1">1</button>
    <button type="button" data-cols="2">2</button><button type="button" data-cols="3">3</button><button type="button" data-cols="4">4</button></div></div>
  <div class="ctl"><span class="ctl-label">Show</span><div class="seg presets" data-chips-for="#grid">{presets}</div></div>
</div>
<div class="chips" data-chips-for="#grid">{chips}</div>
<div class="cmp-grid" id="grid">{"".join(grid)}</div>
{rel_html}
{pager}
"""
        write(DOCS / "compare" / f"{t}.html", page(title, body, depth, "compare/index.html", wide=True,
                                                   desc=f"{title} for all eight systems"))


def build_figures(figs, fimgs) -> None:
    depth = 0
    known = [g for g, _ in GROUPS]
    groups = list(GROUPS) + [(g, g.replace("_", " ").capitalize()) for g in
                             sorted({e["group"] for e in figs} - set(known))]
    toc, sections = [], []
    for g, label in groups:
        items = [e for e in figs if e["group"] == g]
        toc.append(f'<a class="card mini" href="#{esc(g)}"><h3>{esc(label)}</h3><span class="count">{len(items)} figures</span></a>')
        if not items:
            body = '<p class="note">Nothing registered in this group yet — run <code>./update.sh</code>.</p>'
        else:
            wide = g == "overview"
            blocks = "".join(figure_block(e, fimgs[e["file"]], depth, figs, wide=wide) for e in items)
            body = f'<div class="fig-grid{" one" if wide else ""}">{blocks}</div>'
        sections.append(f'<section id="{esc(g)}"><h2>{esc(label)}</h2>'
                        f'<p class="ink2">{esc(GROUP_BLURB.get(g, ""))}</p>{body}</section>')

    # every CSV table under figures/
    rows = []
    for c in sorted(FIGURES.rglob("*.csv")):
        rel = c.relative_to(OVERALL).as_posix()
        try:
            df = pd.read_csv(c)
            shape = f"{len(df):,} × {df.shape[1]}"
            cols = ", ".join(map(str, df.columns[:12])) + (" …" if df.shape[1] > 12 else "")
        except Exception:  # noqa: BLE001
            shape, cols = "–", "unreadable"
        rows.append(f'<tr><td class="path"><a href="{href(depth, overall_path(rel))}">{esc(rel)}</a></td>'
                    f'<td class="num">{shape}</td><td class="num">{human_size(c.stat().st_size)}</td>'
                    f'<td class="small muted">{esc(cols)}</td></tr>')
    tables = ('<div class="table-wrap"><table class="tbl"><thead><tr><th>File</th><th>Rows × cols</th><th>Size</th>'
              f'<th>Columns</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>') if rows else \
        '<p class="muted">No CSV tables yet.</p>'

    body = f"""
<h1>New figures</h1>
<p class="lede">Figures made by the scripts in <code>overall/scripts/</code> from the snapshot in <code>overall/data/</code>.
System colours are fixed: {" ".join(f'<span class="nowrap">{sw(s.key)}{esc(s.label)}</span>' for s in SYSTEMS)}.</p>
<div class="cards toc">{"".join(toc)}<a class="card mini" href="#tables"><h3>Data tables</h3><span class="count">{len(rows)} CSV files</span></a></div>
{"".join(sections)}
<section id="tables"><h2>Data tables</h2>
<p class="ink2">Every CSV written next to the figures (numbers behind the plots).</p>{tables}</section>
"""
    write(DOCS / "figures.html", page("Figures", body, depth, "figures.html",
                                      desc="All new figures grouped by topic"))


def build_data(sds) -> None:
    depth = 0
    count_rows, rows, missing = [], [], []
    tot = {"copied": [0, 0], "too_large": [0, 0], "excluded": [0, 0]}
    for sd in sds:
        c = {"copied": [0, 0], "too_large": [0, 0], "excluded": [0, 0]}
        for f in sd.files:
            st = f.get("status", "?")
            c.setdefault(st, [0, 0])
            c[st][0] += 1
            c[st][1] += f.get("size", 0)
            tot.setdefault(st, [0, 0])
            tot[st][0] += 1
            tot[st][1] += f.get("size", 0)
            if st != "copied":
                missing.append((sd, f))
        count_rows.append(
            f'<tr><td>{sys_chip(sd.s, depth)}</td>' +
            "".join(f'<td class=num>{c[k][0]} <span class="muted small">{human_size(c[k][1])}</span></td>'
                    for k in ("copied", "too_large", "excluded")) +
            f'<td class="path"><a href="{href(depth, f"../data/{sd.s.key}/")}"><code>data/{esc(sd.s.key)}/</code></a></td></tr>')
        rows.append(inventory_rows(sd, depth, with_system=True))
    count_rows.append('<tr class="total"><td>Total</td>' + "".join(
        f'<td class=num>{tot[k][0]} <span class="muted small">{human_size(tot[k][1])}</span></td>'
        for k in ("copied", "too_large", "excluded")) + "<td></td></tr>")

    miss_rows = "".join(
        f'<tr data-status="{esc(f["status"])}" data-system="{esc(sd.s.key)}"><td>{sys_chip(sd.s, depth)}</td>'
        f'<td class="path"><code>{pth(f["path"])}</code></td><td>{status_badge(f["status"])}</td>'
        f'<td class="num">{human_size(f.get("size", 0))}</td>'
        f'<td class="path"><code>{pth(PROJECT / f.get("source", ""))}</code></td></tr>'
        for sd, f in sorted(missing, key=lambda x: (x[1]["status"], x[0].s.key, x[1]["path"])))

    body = f"""
<h1>Data inventory</h1>
<p class="lede">Each system's <code>analysis_continued/</code> folder is mirrored into <code>overall/data/&lt;system&gt;/</code>
by <code>scripts/sync_data.py</code>. <code>data/manifest.json</code> records every file with size, modification time and
(for copied files) sha256, so git shows exactly which results changed between snapshots.</p>
<p class="note"><b>copied</b> — in the snapshot. <b>too large</b> — over 20 MB (trajectory <code>com_frames_nowater.dcd</code>,
<code>pmf_distances.dat</code>, <code>halved/multidihedral_*_half.dat</code>): listed only, read from the source path.
<b>excluded</b> — by rule (<code>*.in *.prmtop *.log *.nc</code>): listed only.</p>

<h2 id="counts">Counts per system</h2>
<div class="table-wrap"><table class="tbl"><thead><tr><th>System</th><th>copied</th><th>too large</th><th>excluded</th><th>Snapshot folder</th></tr></thead>
<tbody>{"".join(count_rows)}</tbody></table></div>

<h2 id="not-copied">Files that live only in the source folders</h2>
{filter_bar("missing", systems=True)}
<div class="table-wrap tall"><table class="tbl" id="missing"><thead><tr><th>System</th><th>File</th><th>Status</th><th>Size</th><th>Original location</th></tr></thead>
<tbody>{miss_rows}</tbody></table></div>

<h2 id="all">All files</h2>
{filter_bar("manifest", systems=True)}
<div class="table-wrap tall"><table class="tbl" id="manifest"><thead><tr><th>System</th><th>Path</th><th>Size</th><th>Status</th><th>sha256</th><th>Source modified</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
"""
    write(DOCS / "data.html", page("Data", body, depth, "data.html", desc="Snapshot file inventory"))


def build_notes(sds) -> None:
    depth = 0
    numbering = "".join(f"<th>{n}</th>" for n in range(1, 17))
    names = ["ACE", "L", "I", "V", "A", "G", "X", "NHE"] * 2
    cells = "".join(f'<td class="{"cap" if v in ("ACE", "NHE") else ""}">{v}</td>' for v in names)
    pos = "".join(f"<td>{p}</td>" for p in (["cap", 1, 2, 3, 4, 5, 6, "cap"] * 2))
    pep = '<td colspan="8">peptide 1</td><td colspan="8">peptide 2</td>'
    sys_rows = "".join(
        f"<tr><td>{sys_chip(s, depth)}</td><td class=mono>{esc(s.key)}</td><td class=mono>{esc(s.x_resname)}</td>"
        f"<td>{esc(s.note)}</td><td class=path><code>{pth(s.analysis_dir.relative_to(PROJECT))}</code></td></tr>"
        for s in SYSTEMS)
    body = f"""
<h1>Methods &amp; notes</h1>
<p class="lede">How the numbers on these pages were made, the conventions to keep in mind when reading the plots,
and the known problems in the source analysis.</p>
<nav class="toc-inline"><a href="#systems">Systems</a> · <a href="#numbering">Residue numbering</a> · <a href="#pmf">PMFs</a> ·
<a href="#halves">Halves</a> · <a href="#wells">Well depths</a> · <a href="#halved">Halved analysis</a> · <a href="#caveats">Caveats</a> ·
<a href="#update">How to update</a> · <a href="#history">Browsing history</a> · <a href="#layout">Folder layout</a></nav>

<h2 id="systems">Systems</h2>
<p>Two LIVAGX peptides per box at 60 mM, parallel start, Amber force field ("new ff"), 300 K, tripled simulation time.
X (position 6) varies; the three LIVAGE systems differ only in Glu protonation.</p>
<div class="table-wrap"><table class="tbl"><thead><tr><th>System</th><th>Key</th><th>Residue 6 (Amber)</th><th>Charge / protonation</th><th>Source folder</th></tr></thead>
<tbody>{sys_rows}</tbody></table></div>

<h2 id="numbering">Residue numbering</h2>
<p>cpptraj residue numbers in the stripped topology (<code>*_nowater.prmtop</code>, residues 1–16 kept by <code>strip !:1-16</code>):</p>
<div class="table-wrap"><table class="tbl numbering"><thead><tr><th>cpptraj :</th>{numbering}</tr></thead>
<tbody><tr><th>residue</th>{cells}</tr><tr><th>position</th>{pos}</tr><tr><th>peptide</th>{pep}</tr></tbody></table></div>
<p>"Pair (i, j)" always means position <i>i</i> on peptide 1 (residue i+1) and position <i>j</i> on peptide 2 (residue j+9),
so the 6×6 grids are not symmetric: (1, 2) and (2, 1) are different distances. In cross-system plots residue 6 is written X
(e.g. L1–X6, window AGX).</p>

<h2 id="pmf">Potentials of mean force</h2>
<ul>
<li>Distances: cpptraj <code>distance</code> between residue masks (centre of mass, minimum image) on
<code>com_frames_nowater.dcd</code> (600,000 frames, peptides only), written to <code>pmf_distances.dat</code> in Å and converted to nm.
36 single-residue pairs, the LIVA fragment (<code>:2-5</code> vs <code>:10-13</code>) and 16 three-residue window distances.</li>
<li>PMF(r) = −k<sub>B</sub>T ln P(r), k<sub>B</sub> = 0.0019872041 kcal mol<sup>−1</sup> K<sup>−1</sup>, T = 300 K.</li>
<li>Bins: {PAIR_BIN} nm for residue pairs (density histogram), {WINDOW_BIN} nm for windows and the LIVA fragment (count histogram
normalised to probability). Empty bins are left empty (NaN), not interpolated.</li>
<li>Reference: every curve is shifted so the populated bin nearest {NORMALIZATION_DISTANCE} nm is zero, and only bins below
{NORMALIZATION_DISTANCE} nm are kept. Negative PMF = more probable than at {NORMALIZATION_DISTANCE} nm = attractive well.</li>
<li>Full trajectory: the first 5% of frames are dropped as equilibration.</li>
<li>Three-residue windows LIV, IVA, VAG, AGX: off-diagonal window pairs pool both orderings (window i on peptide 1 with window j on
peptide 2, and j with i), giving 10 unordered pairs.</li>
</ul>

<h2 id="halves">Halves, mean and uncertainty</h2>
<ul>
<li>trim = 5% of all frames; half 1 = frames [trim, mid), half 2 = [mid + trim, end). For the 600,000-frame trajectories this is
frames 30,001–300,000 and 330,001–600,000 (1-based) — the same ranges the halved analysis uses — 270,000 frames each.</li>
<li>Mean of halves: both half-PMFs are interpolated onto a common grid (from the larger first populated bin to the smaller last bin,
capped at {NORMALIZATION_DISTANCE} nm, same bin width) and averaged. Band = |PMF<sub>1</sub> − PMF<sub>2</sub>| / 2.</li>
<li>Files: <code>single_residue_pmf_halves_value/</code> (both halves), <code>single_residue_pmf_mean_value/</code> (halves + common grid +
mean + band), <code>single_residue_pmf_full_value/</code>, <code>three_residue_pmf_halves_value/</code>, <code>fragment_pmf_halves_value/</code>.</li>
</ul>

<h2 id="wells">Well depths on these pages</h2>
<p>Well depth = minimum of the mean-of-halves PMF over r &lt; {NORMALIZATION_DISTANCE} nm (<code>common.well_depth</code>), reported with the
distance of the minimum and ± |PMF<sub>1</sub>−PMF<sub>2</sub>|/2 at that distance. "Full trajectory" tables use the full-trajectory PMF
(no ±). Because every curve is 0 at {NORMALIZATION_DISTANCE} nm, a depth is always relative to that separation; a value near 0 means no
well deeper than the reference.</p>

<h2 id="halved">Halved structural analysis (<code>halved/</code>)</h2>
<ol>
<li>Frames 30,001–300,000 (half 1) and 330,001–600,000 (half 2) of <code>com_frames_nowater.dcd</code>.</li>
<li>Filter: LIVA–LIVA distance (<code>:2-5</code> vs <code>:10-13</code>) &lt; 7 Å → <code>filtered_frames_*_half.nc</code> (not in the snapshot).
The number of rows in <code>pep1pep2_angle_*_half.dat</code> is therefore the number of contact frames out of 270,000.</li>
<li>On contact frames: <code>multidihedral phi psi resrange 1-16</code> (<code>multidihedral_*_half.dat</code>, too large for the snapshot);
<code>hbond :2-5,10-13 nointramol</code> → <code>interhbond_*_half.txt</code> (inter-peptide H-bonds among L, I, V, A only; G, X and
the caps are not in the mask; Frac = fraction of contact frames); the angle between two vectors → <code>pep1pep2_angle_*_half.dat</code>.</li>
<li>On all frames of each half: end-to-end distance of peptide 1, <code>:2@CA</code>–<code>:7@CA</code> → <code>end_to_end_dist_*_half.dat</code>.</li>
<li>Plots: <code>combined_first_second_half.py</code> (H-bond network and heatmaps, angle and end-to-end histograms).</li>
</ol>
<p>Because H-bond fractions are per contact frame, a system with fewer contacts can show the same fraction with fewer bonds overall;
multiply by the contact fraction for a per-frame comparison (the <code>hbonds_per_all_frames</code> column of
<code>figures/halved/halved_summary.csv</code>).</p>

<h2 id="caveats">Caveats</h2>
<ol class="caveats">
<li><b>Peptide angle vectors.</b> {esc(ANGLE_CAVEAT)} The angle histograms and any angle statistic should be read with this in mind
until the cpptraj input is fixed and re-run.</li>
<li><b>livagr folder name.</b> The source folder is <code>livagr/livagr_tripled_time_new_ff/analysis_contined</code> (typo); the snapshot
normalises it to <code>data/livagr/</code> like every other system, and the manifest keeps the real source path.</li>
<li><b>Halved script label strings.</b> {esc(LABEL_CAVEAT)} In <code>livagh</code> the script now sets
<code>LIVAGH_tripled_time_new_ff_doubled</code>, while the PNGs on disk were written before that edit and are named without
<code>_doubled</code>; re-running the script would create differently named files. These pages strip the label to identify the
plot type, so either spelling is handled.</li>
<li><b>Too-large files.</b> Trajectory, <code>pmf_distances.dat</code> and the multidihedral output are not versioned; only their
size and time stamp are in the manifest. See <a href="data.html#not-copied">Data → files that live only in the source folders</a>.</li>
</ol>

<h2 id="update">How to update</h2>
<pre><code>cd {esc(OVERALL)}
./update.sh</code></pre>
<p><code>update.sh</code> re-syncs the snapshot from the source folders (<code>scripts/sync_data.py</code>, read-only on the source),
regenerates the figures (<code>scripts/fig_*.py</code>), rebuilds this site (<code>scripts/build_docs.py</code>) and commits a git
snapshot with a tag. Only <code>overall/</code> is ever written. To rebuild just the pages:</p>
<pre><code>cd {esc(OVERALL)}
/hopper/home/jzii2024/miniconda3/envs/pmf/bin/python scripts/build_docs.py</code></pre>
<p>Open <code>docs/index.html</code> in a browser (works straight from the file system).</p>

<h2 id="history">Browsing history</h2>
<pre><code>git -C overall log --oneline --decorate        # list snapshots
git -C overall tag                             # list snapshot tags
git -C overall diff --stat &lt;tagA&gt; &lt;tagB&gt; -- data/   # which results changed
git -C overall show &lt;tag&gt;:data/manifest.json   # manifest of an old snapshot
git -C overall checkout &lt;tag&gt;                  # whole folder (docs too) as it was
git -C overall checkout main                   # back to the latest</code></pre>
<p>The <a href="changelog.html">changelog</a> lists every snapshot with the data files that changed per system.</p>

<h2 id="layout">Folder layout of <code>overall/</code></h2>
<ul>
<li><code>data/&lt;system&gt;/</code> — snapshot of each analysis folder; <code>data/manifest.json</code> — every source file with status.</li>
<li><code>figures/</code> — new figures (<code>overview/</code>, <code>cross_system/</code>, <code>halved/</code>, <code>qc/</code>) and their CSV
tables; <code>figures/index.json</code> registers them with title and caption.</li>
<li><code>docs/</code> — this site. <code>docs/systems/previews/</code> holds downscaled WebP copies of the very large original PNGs.</li>
<li><code>scripts/</code> — <code>common.py</code> (systems, loaders), <code>style.py</code> (plot look), <code>sync_data.py</code>,
figure scripts, <code>build_docs.py</code>.</li>
</ul>
"""
    write(DOCS / "notes.html", page("Notes", body, depth, "notes.html", desc="Methods, conventions and caveats"))


def build_changelog(commits, head) -> None:
    depth = 0
    if not commits:
        content = ('<p class="note">No snapshots committed yet. Run <code>./update.sh</code> in <code>overall/</code> '
                   'to record the first one.</p>')
    else:
        order = [s.key for s in SYSTEMS] + ["manifest", "figures", "scripts", "docs", "other"]
        labels = {**{s.key: s.label for s in SYSTEMS}, "manifest": "data/manifest.json", "figures": "figures/",
                  "scripts": "scripts/", "docs": "docs/", "other": "other"}
        verbs = {"A": "added", "M": "modified", "D": "deleted", "R": "renamed", "C": "copied", "T": "type changed"}
        out = []
        for c in commits:
            groups: dict[str, list] = {}
            for st, path, old in c["changes"]:
                groups.setdefault(change_group(path), []).append((st, path, old))
            stat_rows, details = [], []
            for g in sorted(groups, key=lambda g: order.index(g) if g in order else 99):
                items = groups[g]
                cnt = {}
                for st, _, _ in items:
                    cnt[st] = cnt.get(st, 0) + 1
                summary = ", ".join(f"{n} {verbs.get(st, st)}" for st, n in sorted(cnt.items()))
                name = (f"{sw(g)}{esc(labels[g])}" if g in BY_KEY else f"<code>{esc(labels.get(g, g))}</code>")
                stat_rows.append(f"<tr><td>{name}</td><td class=num>{len(items)}</td><td>{esc(summary)}</td></tr>")
                lis = "".join(
                    f'<li><span class="st st-{esc(st)}">{esc(st)}</span> <code>{esc(path)}</code>'
                    f'{f" <span class=muted>← {esc(old)}</span>" if old else ""}</li>' for st, path, old in items[:400])
                more = f"<li class=muted>… {len(items) - 400} more</li>" if len(items) > 400 else ""
                details.append(f'<details><summary>{name} <span class="muted">({len(items)} file{"s" if len(items) != 1 else ""})</span></summary>'
                               f'<ul class="files">{lis}{more}</ul></details>')
            tags = "".join(f'<span class="badge tag">{esc(t)}</span>' for t in c["tags"])
            date = c["date"].replace("T", " ")[:16]
            stats = (f'<div class="table-wrap"><table class="tbl compact"><thead><tr><th>Area</th><th>Files</th><th>Changes</th>'
                     f'</tr></thead><tbody>{"".join(stat_rows)}</tbody></table></div>{"".join(details)}'
                     if stat_rows else '<p class="muted small">No file changes recorded (merge or empty commit).</p>')
            out.append(f'<article class="commit" id="c-{esc(c["short"])}"><div class="head"><time>{esc(date)}</time>'
                       f'<code>{esc(c["short"])}</code>{tags}<span class="muted">{esc(c["author"])}</span></div>'
                       f'<h3>{esc(c["subject"])}</h3>{stats}</article>')
        content = "".join(out)
    head_note = f"HEAD at build time: <code>{esc(head)}</code>. " if head else ""
    body = f"""
<h1>Changelog</h1>
<p class="lede">One entry per git snapshot of <code>overall/</code>, newest first, with the files that changed grouped by system.
{head_note}A snapshot committed by the same <code>update.sh</code> run that built this page appears after the next rebuild.</p>
{content}
"""
    write(DOCS / "changelog.html", page("Changelog", body, depth, "changelog.html", desc="Git snapshot history"))


# ================================================================== assets
CSS = r"""/* LIVAGX docs: generated by overall/scripts/build_docs.py — edit there, not here. */
:root{
  color-scheme:light;
  --page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;--hairline:#e1e0d9;
  --chip:#f0efec;--link:#1c5cab;--focus:#2a78d6;--paper:#ffffff;--bar:#cde2fb;--shadow:0 8px 24px rgba(11,11,11,.10);
  --font:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
/*SYS_LIGHT*/
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){/*DARK_VARS*/}
  :root:not([data-theme="light"]) .heat td.v{background:var(--hd);color:var(--td)}
  :root:not([data-theme="light"]) .legend-bar{background:var(--gd)}
}
:root[data-theme="dark"]{/*DARK_VARS*/}
:root[data-theme="dark"] .heat td.v{background:var(--hd);color:var(--td)}
:root[data-theme="dark"] .legend-bar{background:var(--gd)}

*,*::before,*::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;scroll-padding-top:72px}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 var(--font);overflow-wrap:break-word}
body.lb-open{overflow:hidden}
a{color:var(--link);text-underline-offset:.22em;text-decoration-thickness:1px}
a:has(> code:only-child){text-decoration:none}
a > code{color:var(--link)}
a:hover > code{text-decoration:underline;text-underline-offset:.3em}
a:hover{text-decoration-thickness:2px}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px;border-radius:4px}
img{max-width:100%;height:auto}
code,kbd,pre,.mono{font-family:var(--mono);font-size:.88em}
code{background:var(--chip);padding:.06em .35em;border-radius:5px;overflow-wrap:break-word}
pre{background:var(--surface);border:1px solid var(--hairline);border-radius:10px;padding:12px 14px;overflow-x:auto;line-height:1.5}
pre code{background:none;padding:0;overflow-wrap:normal}
h1,h2,h3{line-height:1.2;letter-spacing:-.01em;margin:0 0 .5em;color:var(--ink)}
h1{font-size:clamp(1.6rem,3.4vw,2.3rem);font-weight:680}
h2{font-size:1.32rem;font-weight:650;margin-top:2.4em}
h3{font-size:1rem;font-weight:620}
p{margin:0 0 .9em}
ul,ol{padding-left:1.3em}
li{margin:.25em 0}
.muted{color:var(--muted)}.ink2{color:var(--ink-2)}.small{font-size:.86em}.nowrap{white-space:nowrap}
small{font-size:.8em;font-weight:500}
.skip{position:absolute;left:-999px;top:0;background:var(--surface);padding:8px 12px;z-index:200}
.skip:focus{left:16px;top:8px}

/* top bar */
.topbar{position:sticky;top:0;z-index:40;background:var(--surface);border-bottom:1px solid var(--hairline)}
.nav{max-width:1600px;margin:0 auto;padding:8px 16px;display:flex;flex-wrap:wrap;align-items:center;gap:4px 4px;position:relative}
.brand{font-weight:700;color:var(--ink);text-decoration:none;margin-right:12px;white-space:nowrap}
.brand span{color:var(--muted);font-weight:500}
.nav a.nl,.dropdown>summary{color:var(--ink-2);text-decoration:none;padding:5px 9px;border-radius:7px;font-size:14px;cursor:pointer;white-space:nowrap;display:inline-block}
.nav a.nl:hover,.dropdown>summary:hover{background:var(--chip);color:var(--ink)}
.nav a.nl[aria-current="page"],.dropdown>summary.cur{background:var(--chip);color:var(--ink);font-weight:600}
.dropdown{position:relative}
.dropdown>summary{list-style:none}
.dropdown>summary::-webkit-details-marker{display:none}
.dropdown>summary::after{content:"▾";margin-left:4px;font-size:.8em;color:var(--muted)}
.dropdown .menu{position:absolute;top:calc(100% + 6px);left:0;min-width:270px;background:var(--surface);border:1px solid var(--hairline);border-radius:10px;padding:6px;box-shadow:var(--shadow);z-index:60}
.dropdown .menu a{display:flex;align-items:center;gap:2px;padding:6px 8px;border-radius:6px;color:var(--ink);text-decoration:none;font-size:14px}
.dropdown .menu a span:first-of-type{flex:1}
.dropdown .menu a:hover,.dropdown .menu a[aria-current]{background:var(--chip)}
.spacer{flex:1}
.theme-toggle{font:inherit;font-size:13px;color:var(--ink-2);background:transparent;border:1px solid var(--hairline);border-radius:999px;padding:4px 11px;cursor:pointer;white-space:nowrap}
.theme-toggle:hover{color:var(--ink);border-color:var(--ink-2)}
@media (max-width:700px){
  .nav{flex-wrap:nowrap;overflow-x:auto;scrollbar-width:none;gap:2px}
  .nav::-webkit-scrollbar{display:none}
  .brand{margin-right:6px}
  .brand span,.spacer{display:none}
  .dropdown{position:static}
  .dropdown .menu{position:fixed;top:46px;left:16px;right:16px;min-width:0;max-height:70vh;overflow:auto}
  .nav a.nl,.dropdown>summary{padding:4px 7px;font-size:13.5px}
  .theme-toggle{margin-left:6px}
}

/* layout */
.container{max-width:1180px;margin:0 auto;padding:28px 16px 72px}
.container.wide{max-width:1600px}
.site-footer{border-top:1px solid var(--hairline);color:var(--muted);font-size:13px;background:var(--surface)}
.site-footer .inner{max-width:1600px;margin:0 auto;padding:18px 16px 32px}
.crumb{font-size:13.5px;color:var(--muted);margin-bottom:6px}
.crumb a{color:var(--ink-2)}
.lede{font-size:1.06rem;color:var(--ink-2);max-width:72ch}
.hero{padding:8px 0 4px}
.eyebrow{font-size:12.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.facts{display:flex;flex-wrap:wrap;gap:10px 28px;margin:18px 0 0;padding:14px 0 0;border-top:1px solid var(--hairline)}
.facts div{min-width:0}
.facts dt{font-size:12px;color:var(--muted)}
.facts dd{margin:2px 0 0;font-size:14px;overflow-wrap:anywhere}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr));gap:20px 28px;align-items:start}
.note{border-left:3px solid var(--hairline);background:var(--surface);padding:10px 14px;border-radius:0 8px 8px 0;margin:12px 0;color:var(--ink-2);font-size:14px}
.note.warn{border-left-color:var(--ink-2)}
.toc-inline{font-size:14px;color:var(--muted);margin:10px 0 0}
.links li{margin:.35em 0}
.pad{padding:0 12px 12px}

/* system identity */
.sw{display:inline-block;width:.78em;height:.78em;border-radius:3px;background:var(--c);margin-right:.45em;vertical-align:-.04em;flex:none}
.syslink{white-space:nowrap;color:var(--ink);text-decoration:none;font-weight:550}
a.syslink:hover{text-decoration:underline}
.sys-head{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px 20px;margin-bottom:12px}
.sys-head h1{margin:0}
.seq{display:flex;flex-wrap:wrap;gap:4px;align-items:center;margin:14px 0 4px}
.chain{display:inline-flex;flex-wrap:nowrap;gap:4px;align-items:center;margin-right:12px}
.pepl{font-size:11px;color:var(--muted);margin-right:2px;font-weight:600}
.res{display:inline-grid;place-items:center;min-width:30px;height:30px;padding:0 6px;border:1px solid var(--hairline);border-radius:7px;background:var(--surface);font-weight:650;font-family:var(--mono);font-size:13px;color:var(--ink)}
.res sub{font-size:9px;color:var(--muted);font-weight:500}
.res.cap{color:var(--muted);font-weight:500;font-size:10.5px}
.res.x{border:2px solid var(--c,var(--ink-2))}

/* cards */
.cards{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(min(100%,250px),1fr));margin-top:10px}
a.card{display:flex;flex-direction:column;text-decoration:none;color:var(--ink);background:var(--surface);border:1px solid var(--hairline);border-radius:12px;padding:16px;transition:border-color .15s}
a.card:hover{border-color:var(--ink-2)}
a.card h3{margin:0 0 6px}
a.card p{margin:0;color:var(--ink-2);font-size:14px;flex:1}
a.card .count{font-size:12.5px;color:var(--muted);margin-top:10px}
a.card.mini{padding:12px 14px}
a.card.mini h3{margin:0}
.pt-thumb{display:block;background:var(--paper);border:1px solid var(--hairline);border-radius:8px;overflow:hidden;margin:-4px -4px 12px;aspect-ratio:4/3}
.pt-thumb img{width:100%;height:100%;object-fit:cover;object-position:top left;display:block}

/* stats */
.stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,215px),1fr));gap:12px}
.stat{background:var(--surface);border:1px solid var(--hairline);border-radius:12px;padding:12px 14px}
.stat .k{font-size:12.5px;color:var(--ink-2)}
.stat .v{font-size:1.3rem;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.3;margin:2px 0}
.stat .v small{font-size:.62em;color:var(--muted);font-weight:500}
.stat .d{font-size:12.5px;color:var(--muted)}

/* figures */
.fig-grid{display:grid;gap:20px;grid-template-columns:repeat(auto-fill,minmax(min(100%,440px),1fr))}
.fig-grid.one{grid-template-columns:minmax(0,1fr)}
.fig{margin:0;background:var(--surface);border:1px solid var(--hairline);border-radius:12px;padding:12px;display:flex;flex-direction:column;gap:10px;min-width:0}
.fig.wide{margin-bottom:20px}
.fig figcaption h3{margin:0 0 4px}
.fig figcaption p{color:var(--ink-2);font-size:14px;margin:0 0 6px}
.fig .meta{font-size:12.5px;color:var(--muted);display:flex;flex-wrap:wrap;gap:4px 14px;align-items:baseline}
.shot{display:block;background:var(--paper);border:1px solid var(--hairline);border-radius:8px;overflow:hidden;cursor:zoom-in}
.shot img{display:block;width:100%;height:auto}

/* compare */
.controls{display:flex;flex-wrap:wrap;gap:10px 22px;align-items:center;margin:16px 0 10px}
.ctl{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.ctl-label{font-size:12.5px;color:var(--muted)}
.seg{display:inline-flex;flex-wrap:wrap;border:1px solid var(--hairline);border-radius:8px;overflow:hidden;background:var(--surface)}
.seg button{font:inherit;font-size:13px;background:var(--surface);color:var(--ink-2);border:0;border-right:1px solid var(--hairline);padding:5px 10px;cursor:pointer}
.seg button:last-child{border-right:0}
.seg button:hover{color:var(--ink)}
.seg button[aria-pressed="true"]{background:var(--chip);color:var(--ink);font-weight:600}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 18px}
.chip{font:inherit;font-size:13px;display:inline-flex;align-items:center;background:var(--surface);color:var(--ink);border:1px solid var(--hairline);border-radius:999px;padding:4px 11px;cursor:pointer}
.chip[aria-pressed="false"]{color:var(--muted)}
.chip[aria-pressed="false"] .sw{background:transparent;box-shadow:inset 0 0 0 1.5px var(--c)}
.cmp-grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(min(100%,380px),1fr))}
.cmp-grid[data-cols="1"]{grid-template-columns:minmax(0,1fr)}
.cmp-grid[data-cols="2"]{grid-template-columns:repeat(2,minmax(0,1fr))}
.cmp-grid[data-cols="3"]{grid-template-columns:repeat(3,minmax(0,1fr))}
.cmp-grid[data-cols="4"]{grid-template-columns:repeat(4,minmax(0,1fr))}
@media (max-width:760px){.cmp-grid[data-cols]{grid-template-columns:minmax(0,1fr)}}
.cmp-card{background:var(--surface);border:1px solid var(--hairline);border-top:3px solid var(--c);border-radius:12px;overflow:hidden;display:flex;flex-direction:column;min-width:0}
.cmp-card[hidden]{display:none}
.cmp-card header{display:flex;gap:4px 10px;align-items:baseline;padding:10px 12px;flex-wrap:wrap}
.cmp-card header .sw{align-self:center}
.cmp-card .shot{margin:0 12px}
.cmp-card footer{padding:8px 12px 12px;font-size:13px;color:var(--muted)}
.pager{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;margin:28px 0 8px}
.sys-head .pager{margin:0}
.pager a{display:inline-flex;align-items:center;padding:6px 12px;border:1px solid var(--hairline);border-radius:999px;background:var(--surface);text-decoration:none;color:var(--ink);font-size:14px}
.pager a:hover{border-color:var(--ink-2)}

/* tables */
.table-wrap{overflow-x:auto;border:1px solid var(--hairline);border-radius:10px;background:var(--surface);margin:8px 0 14px;max-width:100%}
.table-wrap.plain{border:0;background:none}
.table-wrap.tall{max-height:72vh;overflow:auto}
.tbl{border-collapse:collapse;width:100%;font-size:14px}
.tbl th,.tbl td{padding:7px 10px;border-bottom:1px solid var(--hairline);text-align:left;vertical-align:top}
.tbl tbody tr:last-child td,.tbl tbody tr:last-child th{border-bottom:0}
.tbl th{font-weight:600;color:var(--ink-2);font-size:12.5px;background:var(--surface)}
.tbl thead th{position:sticky;top:0;z-index:1}
.tbl td.num,.tbl th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tbl td.path{overflow-wrap:break-word;min-width:10em}
.tbl .src{font-size:12px;color:var(--muted);margin-top:2px}
.tbl a{text-decoration:none}
.tbl a:hover{text-decoration:underline}
.tbl.hb td,.tbl.hb th{white-space:nowrap;padding:6px 7px;font-size:13px}
.tbl.systems{min-width:980px}
.tbl.systems td.path{min-width:15em}
.tbl tr.total td{font-weight:600;border-top:1px solid var(--ink-2)}
.tbl.compact{width:auto;min-width:min(100%,520px)}
.tbl.compact td,.tbl.compact th{padding:4px 12px}
.tbl.numbering td,.tbl.numbering th{text-align:center;padding:5px 6px;font-family:var(--mono);font-size:12.5px}
.tbl.numbering td.cap{color:var(--muted)}
.tbl td.bar{background:linear-gradient(90deg,var(--bar) var(--p),transparent var(--p))}
.pep{display:inline-block;font-size:10.5px;font-weight:700;border:1px solid var(--hairline);border-radius:4px;padding:0 4px;color:var(--ink-2)}
.heat{border-collapse:separate;border-spacing:3px;font-variant-numeric:tabular-nums}
.heat th{font-size:12px;color:var(--ink-2);font-weight:600;padding:2px 6px;text-align:center;font-family:var(--mono)}
.heat th.corner{font-family:var(--font);font-weight:500;color:var(--muted);font-size:10.5px;line-height:1.25;text-align:left}
.heat th.corner span{display:block;white-space:nowrap}
.heat td{border-radius:6px;padding:6px 4px;text-align:center;min-width:56px;font-size:13px;line-height:1.2}
.heat td.v{background:var(--hl);color:var(--tl)}
.heat td.na{background:var(--chip);color:var(--muted)}
.heat td.tail{background:var(--chip);color:var(--muted);font-style:italic}
.heat td.blank{background:none}
.heat td small{display:block;font-size:10.5px;opacity:.82;font-weight:400}
.legend{display:flex;flex-wrap:wrap;align-items:center;gap:6px 18px;font-size:12.5px;margin:6px 0 10px}
.legend-scale{display:inline-flex;flex-direction:column;gap:2px;width:min(240px,60vw)}
.legend-bar{display:block;height:10px;border-radius:5px;background:var(--gl)}
.legend-ticks{display:flex;justify-content:space-between;font-variant-numeric:tabular-nums;color:var(--ink-2);font-size:11.5px}
.tailchip{display:inline-block;background:var(--chip);color:var(--muted);font-style:italic;border-radius:4px;padding:0 5px;margin-right:4px}
.badge{display:inline-block;font-size:11.5px;padding:1px 8px;border-radius:999px;border:1px solid var(--hairline);color:var(--ink-2);background:var(--chip);white-space:nowrap}
.badge.too_large{background:transparent;border-color:var(--ink-2)}
.badge.excluded{background:transparent;color:var(--muted)}
.badge.tag{background:var(--ink);color:var(--page);border-color:var(--ink)}

/* filters */
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0 4px}
.filters input[type=search],.filters select{font:inherit;font-size:14px;color:var(--ink);background:var(--surface);border:1px solid var(--hairline);border-radius:8px;padding:6px 10px;min-width:0;max-width:100%}
.filters input[type=search]{flex:1 1 240px}
.count{font-size:13px;color:var(--muted)}

/* changelog */
.commit{background:var(--surface);border:1px solid var(--hairline);border-radius:12px;padding:14px 16px;margin:14px 0}
.commit .head{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:baseline;font-size:13.5px;color:var(--ink-2)}
.commit h3{margin:6px 0 10px}
.commit details{margin:4px 0}
.commit summary{cursor:pointer;font-size:14px}
.files{list-style:none;padding:4px 0 4px 8px;margin:0;font-size:13px;max-height:320px;overflow:auto}
.st{display:inline-block;width:1.4em;text-align:center;font-family:var(--mono);font-size:11px;border:1px solid var(--hairline);border-radius:4px;color:var(--ink-2)}
.caveats li{margin:.6em 0}

/* lightbox */
.lb-overlay{position:fixed;inset:0;z-index:100;background:rgba(13,13,13,.94);display:flex;flex-direction:column;color:#f0efec}
.lb-overlay[hidden]{display:none}
.lb-bar{display:flex;align-items:center;gap:6px 12px;padding:10px 16px;font-size:14px;flex-wrap:wrap}
.lb-bar .sw{width:12px;height:12px}
.lb-title{font-weight:650}
.lb-sub{color:#c3c2b7;font-size:13px;max-width:60ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lb-bar .grow{flex:1}
.lb-count{color:#c3c2b7;font-variant-numeric:tabular-nums}
.lb-btn{font:inherit;font-size:13px;color:#f0efec;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.2);border-radius:8px;padding:5px 11px;cursor:pointer;text-decoration:none}
.lb-btn:hover{background:rgba(255,255,255,.16)}
.lb-stage{flex:1;min-height:0;overflow:auto;display:flex;align-items:center;justify-content:center;padding:0 60px 16px}
.lb-stage img{max-width:100%;max-height:calc(100vh - 80px);background:#fff;border-radius:6px;cursor:zoom-in;display:block}
.lb-overlay.zoomed .lb-stage{display:block}
.lb-overlay.zoomed .lb-stage img{max-width:none;max-height:none;cursor:zoom-out;margin:0 auto}
.lb-arrow{position:fixed;top:50%;transform:translateY(-50%);width:44px;height:72px;border:0;border-radius:10px;background:rgba(255,255,255,.08);color:#f0efec;font-size:30px;line-height:1;cursor:pointer}
.lb-arrow:hover{background:rgba(255,255,255,.18)}
.lb-arrow[disabled]{display:none}
.lb-prev{left:8px}.lb-next{right:8px}
@media (max-width:700px){
  .lb-stage{padding:0 8px 70px}
  .lb-arrow{top:auto;bottom:10px;transform:none;height:48px}
  .lb-sub{display:none}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = r"""/* LIVAGX docs: generated by overall/scripts/build_docs.py — edit there, not here. */
(function () {
  'use strict';
  var THEME_KEY = '__THEME_KEY__';
  function store(k, v) { try { if (v === null) localStorage.removeItem(k); else localStorage.setItem(k, v); } catch (e) {} }
  function load(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  var root = document.documentElement;

  /* ---- theme: auto -> light -> dark -> auto ---- */
  var tbtn = document.querySelector('[data-theme-toggle]');
  function themeLabel() {
    if (!tbtn) return;
    var t = root.getAttribute('data-theme') || 'auto';
    tbtn.textContent = 'Theme: ' + t;
    tbtn.setAttribute('aria-label', 'Colour theme ' + t + ', click to change');
  }
  if (tbtn) {
    tbtn.addEventListener('click', function () {
      var t = root.getAttribute('data-theme');
      var next = !t ? 'light' : (t === 'light' ? 'dark' : null);
      if (next) root.setAttribute('data-theme', next); else root.removeAttribute('data-theme');
      store(THEME_KEY, next);
      themeLabel();
    });
    themeLabel();
  }

  /* ---- systems dropdown: close on outside click ---- */
  document.addEventListener('click', function (e) {
    document.querySelectorAll('details.dropdown[open]').forEach(function (d) {
      if (!d.contains(e.target)) d.removeAttribute('open');
    });
  });

  /* ---- table filters ---- */
  document.querySelectorAll('.filters[data-table]').forEach(function (bar) {
    var table = document.querySelector(bar.getAttribute('data-table'));
    if (!table || !table.tBodies.length) return;
    var rows = Array.prototype.slice.call(table.tBodies[0].rows);
    var texts = rows.map(function (r) { return r.textContent.toLowerCase(); });
    var q = bar.querySelector('input[type=search]');
    var sels = Array.prototype.slice.call(bar.querySelectorAll('select[data-attr]'));
    var out = bar.querySelector('[data-count]');
    var unit = bar.getAttribute('data-unit') || 'rows';
    function run() {
      var terms = ((q && q.value) || '').toLowerCase().split(/\s+/).filter(Boolean);
      var n = 0;
      rows.forEach(function (r, i) {
        var ok = terms.every(function (t) { return texts[i].indexOf(t) >= 0; });
        sels.forEach(function (s) {
          if (ok && s.value && r.getAttribute('data-' + s.getAttribute('data-attr')) !== s.value) ok = false;
        });
        r.hidden = !ok;
        if (ok) n++;
      });
      if (out) out.textContent = (n === rows.length ? rows.length : n + ' of ' + rows.length) + ' ' + unit;
    }
    if (q) q.addEventListener('input', run);
    sels.forEach(function (s) { s.addEventListener('change', run); });
    run();
  });

  /* ---- compare grid: columns + system chips ---- */
  document.querySelectorAll('[data-cols-for]').forEach(function (seg) {
    var grid = document.querySelector(seg.getAttribute('data-cols-for'));
    if (!grid) return;
    var btns = seg.querySelectorAll('button[data-cols]');
    function set(v) {
      if (v === 'auto') grid.removeAttribute('data-cols'); else grid.setAttribute('data-cols', v);
      btns.forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-cols') === v)); });
    }
    btns.forEach(function (b) {
      b.addEventListener('click', function () { var v = b.getAttribute('data-cols'); set(v); store('livagx-docs-cols', v); });
    });
    var saved = load('livagx-docs-cols');
    set(saved && seg.querySelector('[data-cols="' + saved + '"]') ? saved : 'auto');
  });
  var chipBoxes = document.querySelectorAll('.chips[data-chips-for]');
  chipBoxes.forEach(function (box) {
    var grid = document.querySelector(box.getAttribute('data-chips-for'));
    if (!grid) return;
    var chips = box.querySelectorAll('button[data-key]');
    var presets = document.querySelectorAll('.presets[data-chips-for="' + box.getAttribute('data-chips-for') + '"] button[data-preset]');
    function apply() {
      var on = {};
      chips.forEach(function (c) { on[c.getAttribute('data-key')] = c.getAttribute('aria-pressed') === 'true'; });
      grid.querySelectorAll('.cmp-card[data-key]').forEach(function (card) { card.hidden = !on[card.getAttribute('data-key')]; });
      presets.forEach(function (p) {
        var keys = p.getAttribute('data-preset').split(',');
        var match = Object.keys(on).every(function (k) { return on[k] === (keys.indexOf(k) >= 0); });
        p.setAttribute('aria-pressed', String(match));
      });
    }
    chips.forEach(function (c) {
      c.addEventListener('click', function () {
        c.setAttribute('aria-pressed', c.getAttribute('aria-pressed') === 'true' ? 'false' : 'true');
        apply();
      });
    });
    presets.forEach(function (p) {
      p.addEventListener('click', function () {
        var keys = p.getAttribute('data-preset').split(',');
        chips.forEach(function (c) { c.setAttribute('aria-pressed', String(keys.indexOf(c.getAttribute('data-key')) >= 0)); });
        apply();
      });
    });
    apply();
  });

  /* ---- lightbox: a.lb[data-lb=group]; ←/→ within the visible group, Esc closes ---- */
  var lb, img, ttl, sub, cnt, full, swatch, prevB, nextB, items = [], idx = 0, lastFocus = null;
  function build() {
    lb = document.createElement('div');
    lb.className = 'lb-overlay';
    lb.hidden = true;
    lb.setAttribute('role', 'dialog');
    lb.setAttribute('aria-modal', 'true');
    lb.setAttribute('aria-label', 'Image viewer');
    lb.innerHTML = '<div class="lb-bar"><span class="sw lb-sw"></span><span class="lb-title"></span>' +
      '<span class="lb-sub"></span><span class="grow"></span><span class="lb-count"></span>' +
      '<button type="button" class="lb-btn" data-act="zoom">Zoom</button>' +
      '<a class="lb-btn lb-full" target="_blank" rel="noopener">Full resolution</a>' +
      '<button type="button" class="lb-btn" data-act="close" aria-label="Close viewer">Close ✕</button></div>' +
      '<div class="lb-stage"><img alt=""></div>' +
      '<button type="button" class="lb-arrow lb-prev" data-act="prev" aria-label="Previous image">‹</button>' +
      '<button type="button" class="lb-arrow lb-next" data-act="next" aria-label="Next image">›</button>';
    document.body.appendChild(lb);
    img = lb.querySelector('.lb-stage img');
    ttl = lb.querySelector('.lb-title'); sub = lb.querySelector('.lb-sub'); cnt = lb.querySelector('.lb-count');
    full = lb.querySelector('.lb-full'); swatch = lb.querySelector('.lb-sw');
    prevB = lb.querySelector('.lb-prev'); nextB = lb.querySelector('.lb-next');
    lb.addEventListener('click', function (e) {
      var a = e.target.closest('[data-act]');
      if (a) { act(a.getAttribute('data-act')); return; }
      if (e.target === img) { act('zoom'); return; }
      if (e.target.classList.contains('lb-stage')) close();
    });
  }
  function visible(el) { return !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length); }
  function show() {
    var a = items[idx];
    lb.classList.remove('zoomed');
    img.src = a.getAttribute('href');
    img.alt = a.getAttribute('data-title') || '';
    ttl.textContent = a.getAttribute('data-title') || '';
    sub.textContent = a.getAttribute('data-sub') || '';
    var c = a.getAttribute('data-color');
    swatch.style.display = c ? '' : 'none';
    if (c) swatch.style.setProperty('--c', c);
    full.href = a.getAttribute('data-full') || a.getAttribute('href');
    cnt.textContent = items.length > 1 ? (idx + 1) + ' / ' + items.length : '';
    prevB.disabled = nextB.disabled = items.length < 2;
    [idx - 1, idx + 1].forEach(function (k) {
      var n = items[(k + items.length) % items.length];
      if (n && n !== a) { var im = new Image(); im.src = n.getAttribute('href'); }
    });
  }
  function open(link) {
    if (!lb) build();
    var g = link.getAttribute('data-lb');
    items = Array.prototype.filter.call(document.querySelectorAll('a.lb'), function (a) {
      return a.getAttribute('data-lb') === g && visible(a);
    });
    idx = items.indexOf(link);
    if (idx < 0) { items = [link]; idx = 0; }
    lastFocus = document.activeElement;
    lb.hidden = false;
    document.body.classList.add('lb-open');
    show();
    lb.querySelector('[data-act=close]').focus();
  }
  function close() {
    lb.hidden = true;
    img.removeAttribute('src');
    document.body.classList.remove('lb-open');
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }
  function act(a) {
    if (a === 'close') close();
    else if (a === 'prev' && items.length > 1) { idx = (idx - 1 + items.length) % items.length; show(); }
    else if (a === 'next' && items.length > 1) { idx = (idx + 1) % items.length; show(); }
    else if (a === 'zoom') lb.classList.toggle('zoomed');
  }
  document.addEventListener('click', function (e) {
    var a = e.target.closest ? e.target.closest('a.lb') : null;
    if (!a || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    open(a);
  });
  document.addEventListener('keydown', function (e) {
    if (!lb || lb.hidden) {
      if (e.key === 'Escape') document.querySelectorAll('details.dropdown[open]').forEach(function (d) { d.removeAttribute('open'); });
      return;
    }
    if (e.key === 'Escape') { e.preventDefault(); close(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); act('prev'); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); act('next'); }
  });
})();
"""


def build_assets() -> None:
    light = "\n".join(f"  --sys-{s.key}:{s.color};" for s in SYSTEMS)
    dark_sys = "".join(f"--sys-{s.key}:{PALETTE_DARK[k]};" for k, s in enumerate(SYSTEMS))
    dark = ("color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#8f8d86;"
            "--hairline:#2c2c2a;--chip:#242422;--link:#86b6ef;--focus:#3987e5;--paper:#ffffff;--bar:#184f95;"
            "--shadow:0 8px 24px rgba(0,0,0,.5);" + dark_sys)
    css = CSS.replace("/*SYS_LIGHT*/", light).replace("/*DARK_VARS*/", dark)
    write(DOCS / "assets" / "site.css", css)
    write(DOCS / "assets" / "site.js", JS.replace("__THEME_KEY__", THEME_KEY))


# ================================================================== main
def clean_stale(folder: Path, keep: set[Path]) -> None:
    if not folder.exists():
        return
    for f in folder.glob("*.html"):
        if f.resolve() not in keep:
            assert_inside_overall(f).unlink()
            print(f"removed stale {f.relative_to(OVERALL)}")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    figs = load_figure_index()
    prev = Previewer()
    if Image is None:
        print("NOTE: Pillow not importable - pages will load the original (very large) PNGs")
    sds = collect(manifest, prev)
    fimgs = figure_images(figs, prev)
    prev.finish()
    commits, head = git_history()

    vmax_pair = nice_max([v[0] for sd in sds for d in (sd.pair_mean, sd.pair_full) for v in d.values()])
    vmax_win = nice_max([v[0] for sd in sds for v in list(sd.windows.values()) + [sd.fragment]])

    build_assets()
    build_index(sds, figs, fimgs, commits, head, manifest)
    build_system_pages(sds, figs, fimgs, vmax_pair, vmax_win)
    build_compare(sds, figs)
    build_figures(figs, fimgs)
    build_data(sds)
    build_notes(sds)
    build_changelog(commits, head)

    keep = {p.resolve() for p in WRITTEN}
    clean_stale(DOCS / "systems", keep)
    clean_stale(DOCS / "compare", keep)
    print(f"wrote {len(WRITTEN)} files under {DOCS.relative_to(OVERALL)}/ "
          f"({prev.made} new previews, {len(figs)} registered figures, {len(commits)} commits)")


if __name__ == "__main__":
    main()
