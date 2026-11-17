# GeoVenture AI 3.1 — reproducible historical backtest

Run: `./run_all.sh` (uses data/) or `./run_all.sh --rebuild` (re-downloads the public dump and re-parses it).
Outputs in results/: report.html, metrics.json, predictions.csv, years.csv, scenario_grid.csv.
Check reproduction: `sha256sum -c RESULTS_SHA256SUMS` after the run.

Data: NPD (now Sodir) FactPages, snapshot July 2015, from the public GitHub repo ontop/npd-benchmark
(file data/mysql/original_npd/dropped/npd_july2015.sql). Norwegian Licence for Open Government Data (NLOD).
To use a current Sodir export, replace data/*.csv with FactPages CSVs (same attribute names: wlbWellboreName, wlbContent, ...).

Pre-registered design: see the header of scripts/backtest31.py.
Requires: Python 3.10+, pandas, numpy, scikit-learn, matplotlib.
