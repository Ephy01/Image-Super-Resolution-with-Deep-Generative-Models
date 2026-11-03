#!/usr/bin/env bash
# Render a Markdown report to A4 PDF in the format of the stage 2-4 reports (pandoc -> HTML -> headless Chrome).
#   bash scripts/render_report.sh reports/stage5_final.md        # -> reports/stage5_final.pdf
set -euo pipefail
cd "$(dirname "$0")/.."
SOURCE="$1"
HTML="${SOURCE%.md}.html"
PDF="${SOURCE%.md}.pdf"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
pandoc "$SOURCE" --from gfm --to html5 --standalone --css assets/report.css \
  --variable pagetitle="$(basename "${SOURCE%.md}")" -o "$HTML"
"$CHROME" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/$PDF" "file://$PWD/$HTML" 2>/dev/null
rm "$HTML"
echo "$PDF"
