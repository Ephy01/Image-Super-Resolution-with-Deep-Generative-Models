#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

fail() {
    echo "$1"
    if [ -t 0 ]; then
        read -r -p "Press Enter to close. " || true
    fi
    exit 1
}

if [ ! -x .venv/bin/python ]; then
    fail "Run setup.command once before starting the demo."
fi
if ! .venv/bin/python -c 'import platform, sys, torch; raise SystemExit(not (sys.platform == "darwin" and platform.machine() == "arm64" and torch.backends.mps.is_available()))'; then
    fail "Apple GPU (MPS) is unavailable in this environment. Run setup.command and check its output."
fi

echo "Starting the demo. Leave this Terminal window open. Press Ctrl+C to stop."
exec .venv/bin/python -B app.py --device mps --open "$@"
