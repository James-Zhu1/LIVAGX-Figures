# LIVAGX overall documentation

Cross-system documentation of the 60 mM parallel two-peptide LIVAGX runs
(`livag*/*_tripled_time_new_ff/analysis_continued/`). Only this folder is
written; the source analysis folders are read, never changed.

## Browse

Open `docs/index.html` in a browser (copy `overall/` to your laptop or use any
file viewer; the site is self-contained, no internet needed).

| Page | What it shows |
|---|---|
| `docs/index.html` | systems table + the big overview grid (row = peptide, column = PMF family) |
| `docs/systems/<key>.html` | every original plot of one system, well-depth table, top H-bonds, file inventory |
| `docs/compare/` | the same original plot side by side for all 8 systems |
| `docs/figures.html` | new cross-system, halved-structure and convergence figures |
| `docs/explorer.html` | interactive PMF overlay (pick a pair/window, toggle systems, hover values) |
| `docs/data.html` | every source file: copied, too large, or excluded |
| `docs/notes.html` | methods, conventions and known caveats |
| `docs/changelog.html` | git snapshot history |

## Update and version

```bash
./update.sh "what changed"
```

Re-syncs `data/`, regenerates `figures/` and `docs/`, then commits and tags
`snapshot-YYYY-MM-DD_HHMM`. `./update.sh --push "msg"` also pushes if an
`origin` remote is set. History: `git log --stat`, `git checkout snapshot-...`.

## Layout

- `scripts/common.py` system registry (labels, paths, fixed colours), PMF loaders
- `scripts/style.py` shared figure style
- `scripts/sync_data.py` snapshot to `data/<key>/` (skips `*.in *.prmtop *.log *.nc` and files > 20 MB; all listed in `data/manifest.json`)
- `scripts/fig_overview_grid.py` (all systems overlaid), `fig_system_rows.py` (8 rows x 7 analyses), `fig_cross_system.py`, `fig_halved_compare.py`, `fig_qc.py` figures + CSV tables in `figures/`
- `scripts/export_explorer_data.py`, `scripts/build_docs.py` the docs site

To add a system, add one line to `_SYS` in `scripts/common.py` and run `./update.sh`.
