#!/usr/bin/env bash
# Download the pretrained weights (Real-ESRNet and Real-ESRGAN 64 MB each, ResShift 456 MB, autoencoder 211 MB),
# verify them against scripts/SHA256SUMS and check out the pinned official ResShift code.
# Run inside the activated environment. Target: $SR_WEIGHTS or models/; ResShift code: third_party/ResShift.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
"$PY" code/datasets/download_data.py --only weights
bash code/models/setup_resshift.sh
