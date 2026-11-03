#!/usr/bin/env bash
# Download the DIV2K validation images (449 MB) and the 120 FFHQ faces chosen with seed 42, then build the dataset.
# Run inside the activated environment. Target: $SR_DATA or data/ (see paths.py).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
"$PY" code/datasets/download_data.py --only data
"$PY" code/datasets/prepare_data.py
echo "Dataset manifest: ${SR_DATA:-data}/processed/manifest.csv; input hashes: ${SR_DATA:-data}/processed/inputs_sha256.json"
