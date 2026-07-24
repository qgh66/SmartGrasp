#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$HOME/anaconda3/envs/smartgrasp/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "SmartGrasp Python not found: $PYTHON" >&2
    exit 1
fi

# Reuse the existing pipeline's API defaults without copying credentials into
# rsr/. Only these two trusted export assignments are evaluated; no pipeline
# stages are sourced or executed.
while IFS= read -r line; do
    case "$line" in
        "export OPENAI_API_KEY="*)
            [[ -z "${OPENAI_API_KEY:-}" ]] && eval "$line"
            ;;
        "export OPENAI_BASE_URL="*)
            [[ -z "${OPENAI_BASE_URL:-}" ]] && eval "$line"
            ;;
    esac
done < "$ROOT_DIR/run_pipeline.sh"

exec "$PYTHON" -u -m rsr.run "$@"
