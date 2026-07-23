#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$HOME/anaconda3/envs/smartgrasp/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-rsr/data/gpt4o_first10_four_categories/output}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MAIN_PATTERN="[b]ash rsr/run_gpt4o_remaining_four_categories.sh"

if [[ ! -x "$PYTHON" ]]; then
    echo "SmartGrasp Python not found: $PYTHON" >&2
    exit 1
fi
: "${OPENAI_API_KEY:?Please export OPENAI_API_KEY before starting retries}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://yunwu.ai/v1}"

main_pid="$(pgrep -fo "$MAIN_PATTERN" || true)"
if [[ -n "$main_pid" ]]; then
    echo "[wait] main experiment pid=$main_pid"
    while kill -0 "$main_pid" 2>/dev/null; do
        sleep "$POLL_SECONDS"
    done
    echo "[wait] main experiment finished"
else
    echo "[wait] no active main experiment; retrying incomplete tasks now"
fi

mkdir -p "$OUTPUT_ROOT"
retry_log="$OUTPUT_ROOT/retry_failed_$(date +%Y%m%d_%H%M%S).log"

set +e
"$PYTHON" -m rsr.retry_failed_reason --evaluate \
    2>&1 | tee "$retry_log"
retry_status=${PIPESTATUS[0]}
set -e

echo "[retry] log -> $retry_log"
exit "$retry_status"
