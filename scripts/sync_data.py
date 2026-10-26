#!/usr/bin/env python3
"""Snapshot each system's analysis_continued folder into overall/data/<key>/.

Copied:   every file except *.in, *.prmtop, *.log, *.nc, and except files
          larger than MAX_COPY_BYTES (the .dcd trajectory, pmf_distances.dat,
          multidihedral_*.dat).
Recorded: every file, copied or not, goes in data/manifest.json with its size,
          mtime and (for copied files) sha256, so the git history shows exactly
          what changed in the source folders between snapshots.

The source folders are only read. Files that disappeared from a source folder
are removed from the snapshot so the snapshot mirrors the source.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from common import (DATA, EXCLUDE_SUFFIXES, MANIFEST, MAX_COPY_BYTES, PROJECT,
                    SYSTEMS, assert_inside_overall)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sync_system(system) -> list[dict]:
    src_root = system.analysis_dir
    dst_root = assert_inside_overall(system.snapshot)
    entries, kept = [], set()
    for src in sorted(p for p in src_root.rglob("*") if p.is_file()):
        rel = src.relative_to(src_root)
        st = src.stat()
        entry = dict(path=str(rel), size=st.st_size,
                     mtime=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                     source=str(src.relative_to(PROJECT)))
        if src.suffix in EXCLUDE_SUFFIXES:
            entry["status"] = "excluded"
        elif st.st_size > MAX_COPY_BYTES:
            entry["status"] = "too_large"
        else:
            dst = dst_root / rel
            if not (dst.exists() and dst.stat().st_size == st.st_size
                    and int(dst.stat().st_mtime) == int(st.st_mtime)):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, assert_inside_overall(dst))
            entry["status"] = "copied"
            entry["sha256"] = sha256(dst)
            kept.add(dst.resolve())
        entries.append(entry)
    # drop snapshot files whose source is gone (or became excluded / too large)
    if dst_root.exists():
        for old in dst_root.rglob("*"):
            if old.is_file() and old.resolve() not in kept:
                assert_inside_overall(old).unlink()
    return entries


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for s in SYSTEMS:
        entries = sync_system(s)
        manifest[s.key] = dict(label=s.label, sequence=s.sequence, x_resname=s.x_resname,
                               note=s.note, source_dir=str(s.analysis_dir.relative_to(PROJECT)),
                               files=entries)
        n = {k: sum(e["status"] == k for e in entries) for k in ("copied", "too_large", "excluded")}
        print(f"{s.key:10s} copied {n['copied']:4d}  too_large {n['too_large']:2d}  excluded {n['excluded']:2d}")
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
