#!/usr/bin/env bash
# GeoVenture AI 3.1 - full reproducible run.
#   ./run_all.sh            -> uses the CSVs in data/ (parsed from the public NPD/Sodir July-2015 dump)
#   ./run_all.sh --rebuild  -> re-downloads the dump from GitHub (ontop/npd-benchmark) and re-parses it first
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" == "--rebuild" ]]; then
  rm -rf _src && git clone -q --depth 1 --filter=blob:none --no-checkout https://github.com/ontop/npd-benchmark _src
  (cd _src && git checkout HEAD -- data/mysql/original_npd/dropped/npd_july2015.sql)
  python3 -I scripts/load_dump.py _src/data/mysql/original_npd/dropped/npd_july2015.sql data
fi
sha256sum -c DATA_SHA256SUMS
python3 -I scripts/features.py data 365 data/wf_lag1y.csv > /dev/null
python3 -I scripts/backtest31.py data data/wf_lag1y.csv results > results/run.log
python3 -I -c "import json;m=json.load(open('results/metrics.json'));print('n_test',m['n_test'],'| main money',{k:v['value_musd'] for k,v in m['main_money']['strategies'].items()})"
