#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

if [ ! -f "$ROOT/model-router.json" ]; then
  echo "ERROR: model-router.json not found."
  echo "Copy config/model-router.example.json to model-router.json and fill in your model paths."
  exit 1
fi

cd "$ROOT"
exec python -m uvicorn model_router.main:app \
  --host "${BUBBLE_HOST:-127.0.0.1}" \
  --port "${BUBBLE_PORT:-8090}" \
  "$@"
