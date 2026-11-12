#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

finish() {
    status=$?
    if [ "$status" -ne 0 ]; then
        echo
        echo "Setup did not finish. Read the message above, fix the problem, then run setup.command again."
    fi
    if [ -t 0 ]; then
        read -r -p "Press Enter to close. " || true
    fi
}
trap finish EXIT

if [ "$(uname -s)" != "Darwin" ]; then
    echo "This launcher is for macOS on Apple Silicon."
    exit 1
fi
if [ "$(sw_vers -productVersion | cut -d. -f1)" -lt 14 ]; then
    echo "macOS 14 or newer is required by the pinned PyTorch version."
    exit 1
fi
if ! command -v git >/dev/null 2>&1; then
    echo "Git is required to download the pinned ResShift source. Install Git, then run this setup again."
    exit 1
fi

valid_python() {
    "$1" -c 'import platform, sys, ssl, pyexpat, venv, ensurepip; raise SystemExit(not (sys.platform == "darwin" and platform.machine() == "arm64" and (3, 12) <= sys.version_info[:2] <= (3, 14) and sys.version_info[:3] != (3, 14, 1)))' >/dev/null 2>&1
}

DEMO_PY=""
if [ -n "${DEMO_PYTHON:-}" ]; then
    DEMO_PY="$(command -v "$DEMO_PYTHON" || true)"
    if [ -z "$DEMO_PY" ] || ! valid_python "$DEMO_PY"; then
        echo "DEMO_PYTHON must point to a native ARM64 Python 3.12-3.14 on macOS (except 3.14.1). Python 3.12 is recommended."
        exit 1
    fi
else
    for candidate in python3.12 python3.13 python3.14 python3; do
        candidate_path="$(command -v "$candidate" || true)"
        if [ -n "$candidate_path" ] && valid_python "$candidate_path"; then
            DEMO_PY="$candidate_path"
            break
        fi
    done
fi
if [ -z "$DEMO_PY" ]; then
    echo "Install native ARM64 Python 3.12 for macOS, then run this setup again."
    echo "Python downloads: https://www.python.org/downloads/macos/"
    echo "If needed: DEMO_PYTHON=/path/to/python3.12 bash setup.command"
    exit 1
fi

echo "Using $DEMO_PY"
if [ ! -e .venv ]; then
    "$DEMO_PY" -m venv .venv
elif [ ! -x .venv/bin/python ] || ! valid_python .venv/bin/python; then
    echo "The existing .venv is not a supported native ARM64 Python environment."
    echo "Move that .venv folder aside, then run setup.command again."
    exit 1
fi
if ! .venv/bin/python -c 'from pathlib import Path; import sys; raise SystemExit(Path(sys.prefix).resolve() != (Path.cwd() / ".venv").resolve())'; then
    echo "The .venv interpreter does not belong to this demo folder. Move .venv aside and retry."
    exit 1
fi

if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    .venv/bin/python -m ensurepip --upgrade
fi
.venv/bin/python -m pip --disable-pip-version-check --require-virtualenv install -r requirements.txt
.venv/bin/python -B setup_models.py
.venv/bin/python -B scripts/check_installation.py --device mps

echo
echo "Setup is complete. Run start.command to open the demo."
