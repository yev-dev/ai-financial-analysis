#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
DOWNLOADER_PATH="$PROJECT_DIR/scripts/web_page_download.py"

if [ -n "${WEB_PAGE_DOWNLOADER_PYTHON:-}" ]; then
    PYTHON_CMD="$WEB_PAGE_DOWNLOADER_PYTHON"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "Python was not found. Set WEB_PAGE_DOWNLOADER_PYTHON or activate an environment first." >&2
    exit 1
fi

DEFAULT_URL="${WEB_PAGE_DOWNLOAD_URL:-https://www.google.com}"
DEFAULT_OUTPUT_DIR="${WEB_PAGE_DOWNLOAD_OUTPUT_DIR:-$PROJECT_DIR/downloads/google}"

cd "$PROJECT_DIR"

if [ "$#" -eq 0 ]; then
    set -- \
        "$DEFAULT_URL" \
        --selector "a[href]" \
        --pattern "\\.(html|pdf|csv)$" \
        --same-domain \
        --output-dir "$DEFAULT_OUTPUT_DIR" \
        --limit 5 \
        --save-page
fi

exec "$PYTHON_CMD" "$DOWNLOADER_PATH" "$@"