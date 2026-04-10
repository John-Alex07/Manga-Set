#!/usr/bin/env bash
set -euo pipefail

RUNTIME_PYTHON="${RUNTIME_PYTHON:-/tmp/manga-set-venv/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -x "$RUNTIME_PYTHON" ]]; then
  echo "Runtime python not found: $RUNTIME_PYTHON" >&2
  exit 1
fi

cd "$ROOT_DIR"
exec "$RUNTIME_PYTHON" "$@"
