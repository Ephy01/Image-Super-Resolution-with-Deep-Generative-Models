set -euo pipefail
cd "$(dirname "$0")"
# Local dev pipeline with the frozen settings. ResShift on a CPU or Apple GPU is slow; the reported ResShift
# runs and all T4 timings come from the notebooks in experiments/ (see README.md).
PY=${PYTHON:-.venv/bin/python}
CONFIG=${CONFIG:-configs/final.yaml}
$PY prepare.py --config "$CONFIG"
$PY inference.py --config "$CONFIG" --split dev
$PY evaluate.py --config "$CONFIG" --split dev
$PY analyze.py --config "$CONFIG" --split dev
echo "Results: results/final/summary_dev.csv, results/analysis/, results/final/figures/"
