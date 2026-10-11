set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-.venv/bin/python}
$PY code/datasets/download_data.py
$PY code/datasets/prepare_data.py
$PY code/models/upscale.py --split dev
$PY code/evaluation/evaluate.py --split dev
$PY code/evaluation/make_figures.py --split dev
echo "Results: results/summary_dev.csv, results/figures/"
