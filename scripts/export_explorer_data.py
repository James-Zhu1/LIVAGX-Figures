#!/usr/bin/env python3
"""Export every PMF curve of the eight LIVAGX systems for the interactive explorer.

Writes docs/assets/pmf_data.js, a plain script that sets ``window.PMF_DATA``.
The explorer page (docs/explorer.html) is opened straight from disk (file://),
where fetch()/XHR of local JSON is blocked, so the data ships as a .js file.

What is exported, per system (read from the overall/data snapshot via common.py):
  pairs     36 ordered (i, j) single-residue pairs: mean of halves + |PMF1-PMF2|/2
            band, both halves, and the whole-trajectory ("full") PMF
  windows   10 unordered three-residue window pairs (LIV, IVA, VAG, AGX; off-diagonal
            ones pool both orderings): mean + band + both halves (no whole-trajectory
            file exists)
  fragment  LIVA-LIVA: mean + band + both halves
and, for every curve, the well depth (minimum PMF, kcal/mol) and r_min (nm).

Compact curve encoding (keeps the file well under 3 MB):
  {"r0": first distance, "dr": spacing, "y": [...]}   when the grid is uniform
  {"r": [...], "y": [...]}                             otherwise
  mean curves also carry "b": the |PMF1-PMF2|/2 band on the same grid.
PMF values are rounded to 4 decimals (0.0001 kcal/mol); empty bins are null.

Run from overall/:
    /hopper/home/jzii2024/miniconda3/envs/pmf/bin/python scripts/export_explorer_data.py
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

import common
from common import (BY_KEY, DOCS, MANIFEST, NORMALIZATION_DISTANCE, PAIR_BIN, SYSTEMS,
                    WINDOW_BIN, WINDOW_PAIRS, WINDOW_STARTS, assert_inside_overall,
                    load_fragment, load_pair, load_window, pair_ids)

try:  # style.py imports matplotlib; only the dark palette is needed here
    from style import PALETTE_DARK
except Exception:  # pragma: no cover - fall back to the documented values
    PALETTE_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                    "#d55181", "#008300", "#9085e9", "#e66767"]

OUT = DOCS / "assets" / "pmf_data.js"
Y_DECIMALS = 4
R_DECIMALS = 6
MAX_BYTES = 3 * 1024 * 1024


# ------------------------------------------------------------------ encoding
def _num(x, nd=Y_DECIMALS):
    """Round to nd decimals; NaN/inf -> None (JSON null). -0.0 -> 0."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    v = round(x, nd)
    return 0 if v == 0 else v


def _trim(r: np.ndarray, y: np.ndarray, *extra: np.ndarray):
    """Drop leading/trailing points whose distance or value is NaN."""
    ok = np.isfinite(r) & np.isfinite(y)
    if not ok.any():
        return (np.array([]), np.array([])) + tuple(np.array([]) for _ in extra)
    idx = np.flatnonzero(ok)
    sl = slice(idx[0], idx[-1] + 1)
    r, y = r[sl], y[sl]
    extra = tuple(e[sl] for e in extra)
    # interior points with NaN distance cannot be placed: drop them
    keep = np.isfinite(r)
    return (r[keep], y[keep]) + tuple(e[keep] for e in extra)


def encode(r, y, band=None) -> dict | None:
    r = np.asarray(r, float)
    y = np.asarray(y, float)
    arrays = (r, y) if band is None else (r, y, np.asarray(band, float))
    trimmed = _trim(*arrays)
    r, y = trimmed[0], trimmed[1]
    if r.size == 0:
        return None
    out: dict = {}
    if r.size >= 2:
        dr = float(np.median(np.diff(r)))
        recon = r[0] + dr * np.arange(r.size)
        uniform = dr > 0 and float(np.max(np.abs(recon - r))) < 5e-6
    else:
        dr, uniform = 0.0, True
    if uniform:
        out["r0"] = _num(r[0], R_DECIMALS)
        out["dr"] = _num(dr, R_DECIMALS)
    else:
        out["r"] = [_num(v, R_DECIMALS) for v in r]
    out["y"] = [_num(v) for v in y]
    if band is not None:
        out["b"] = [_num(v) for v in trimmed[2]]
    return out


def well(r, y, band=None) -> list | None:
    """[depth kcal/mol, r_min nm, band at r_min or None]; same rule as
    common.well_depth (global minimum of the curve)."""
    r = np.asarray(r, float)
    y = np.asarray(y, float)
    ok = np.isfinite(r) & np.isfinite(y)
    if not ok.any():
        return None
    k = int(np.flatnonzero(ok)[np.argmin(y[ok])])
    b = None if band is None else _num(np.asarray(band, float)[k])
    return [_num(y[k]), _num(r[k], 4), b]


def _halves(c: common.Curve) -> dict:
    out = {}
    for name, h in (("h1", c.half1), ("h2", c.half2)):
        if h is not None:
            out[name] = encode(*h)
    return out


def curve_block(mean_c: common.Curve, full_c: common.Curve | None = None) -> dict:
    blk = {"mean": encode(mean_c.r, mean_c.pmf, mean_c.err)}
    blk.update(_halves(mean_c))
    wells = {"mean": well(mean_c.r, mean_c.pmf, mean_c.err)}
    for name, h in (("h1", mean_c.half1), ("h2", mean_c.half2)):
        if h is not None:
            wells[name] = well(*h)
    if full_c is not None:
        blk["full"] = encode(full_c.r, full_c.pmf)
        wells["full"] = well(full_c.r, full_c.pmf)
    blk["well"] = wells
    return blk


# ------------------------------------------------------------------ export
def data_as_of() -> dict:
    """Newest source mtime of the PMF files per system (from the manifest)."""
    out = {}
    try:
        man = json.loads(MANIFEST.read_text())
    except Exception:
        return out
    for key, ent in man.items():
        times = [f.get("mtime", "") for f in ent.get("files", [])
                 if "_PMF/" in f.get("path", "") and f.get("path", "").endswith(".csv")]
        if times:
            out[key] = max(times)
    return out


def export() -> dict:
    as_of = data_as_of()
    systems = []
    pmf = {}
    for k, s in enumerate(SYSTEMS):
        # residue-6 Amber name on peptide 1 and on peptide 2 ("GLU/GLH" -> both);
        # only LIVAGE (E-/E0) differs: GLU on peptide 1, GLH on peptide 2 (prmtop)
        xs = s.x_resname.split("/")
        systems.append(dict(key=s.key, label=s.label, sequence=s.sequence,
                            x_resname=s.x_resname, x_by_peptide=[xs[0], xs[-1]], note=s.note,
                            color_light=s.color, color_dark=PALETTE_DARK[k],
                            source_dir=str(s.analysis_dir.relative_to(common.PROJECT)),
                            data_as_of=as_of.get(s.key)))
        pairs = {}
        for i, j in pair_ids():
            mean_c = load_pair(s, i, j, "mean")
            full_c = load_pair(s, i, j, "full")
            pairs[f"{i}-{j}"] = curve_block(mean_c, full_c)
        windows = {}
        for a, b in WINDOW_PAIRS:
            windows[f"{a}-{b}"] = curve_block(load_window(s, a, b))
        fragment = curve_block(load_fragment(s))
        pmf[s.key] = dict(pairs=pairs, windows=windows, fragment=fragment)
        print(f"  {s.key:10s} 36 pairs, {len(windows)} windows, fragment")

    # Deterministic stamp (newest source PMF file), so re-running the export on
    # unchanged data gives a byte-identical file and a clean git diff.
    stamp = max(as_of.values()) if as_of else datetime.now().isoformat(timespec="seconds")
    return dict(
        version=1,
        data_as_of=stamp,
        generator="overall/scripts/export_explorer_data.py",
        conventions=dict(
            normalization_nm=NORMALIZATION_DISTANCE,
            pair_bin_nm=PAIR_BIN,
            window_bin_nm=WINDOW_BIN,
            pmf="-kT ln P(r) of the raw centre-of-mass distance histogram at 300 K "
                "(no r^2 / Jacobian volume correction), as in pmf_from_cpptraj.py",
            mean="average of trajectory halves 1 and 2 (the first 5% of all frames dropped "
                 "at the start of each half) on a common grid; band = |PMF1 - PMF2| / 2",
            full="whole trajectory after dropping the first 5% of frames "
                 "(single-residue pairs only)",
            zero="every curve is shifted to 0 at the populated bin nearest 1.8 nm and cut "
                 "there; negative = more probable than at 1.8 nm (attractive)",
            pair="(i, j) = position i on peptide 1 vs position j on peptide 2 "
                 "(residue centre-of-mass distance)",
        ),
        positions=list("LIVAGX"),
        window_starts=WINDOW_STARTS,
        window_names=[common.window_name(None, a) for a in WINDOW_STARTS],
        systems=systems,
        pmf=pmf,
    )


def main() -> None:
    print("exporting PMF curves for the explorer ...")
    data = export()
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    out = assert_inside_overall(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ("/* LIVAGX PMF explorer data - generated by overall/scripts/export_explorer_data.py "
              f"(newest source PMF file: {data['data_as_of']}). Do not edit by hand; re-run the script. */\n")
    out.write_text(header + "window.PMF_DATA = " + text + ";\n", encoding="utf-8")
    size = out.stat().st_size
    print(f"wrote {out} ({size / 1024:.0f} KiB)")
    if size > MAX_BYTES:
        raise SystemExit(f"pmf_data.js is {size} bytes, above the {MAX_BYTES} byte budget")
    assert set(data["pmf"]) == set(BY_KEY)


if __name__ == "__main__":
    main()
