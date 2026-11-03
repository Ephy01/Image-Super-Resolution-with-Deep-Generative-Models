#!/usr/bin/env bash
# Environment setup (macOS / Linux). Run once:
#   ./setup.sh
# Creates a local .venv and installs requirements.txt, then checks for Tesseract.
# Data and weights are downloaded separately: bash scripts/download_data.sh && bash scripts/download_weights.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Python 3.11+ is required (tested with 3.13.9)." >&2
  exit 1
fi
"$PY" - <<'PYCHK' || { echo "ERROR: Python 3.11+ is required." >&2; exit 1; }
import sys

raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PYCHK

echo ">> creating the virtual environment .venv"
"$PY" -m venv "$HERE/.venv"
VPY="$HERE/.venv/bin/python"

echo ">> upgrading pip and installing dependencies"
"$VPY" -m pip install --upgrade pip >/dev/null
"$VPY" -m pip install -r "$HERE/requirements.txt"

if command -v tesseract >/dev/null 2>&1; then
  echo ">> $(tesseract --version 2>&1 | head -1) found (the results used 5.5.1 with the tessdata_fast eng model)"
else
  echo "WARNING: tesseract not found; the OCR metric needs it. macOS: brew install tesseract; Debian/Ubuntu: sudo apt-get install tesseract-ocr" >&2
fi

echo
echo "Done. Next: source .venv/bin/activate && bash scripts/download_data.sh && bash scripts/download_weights.sh, then see README.md."
