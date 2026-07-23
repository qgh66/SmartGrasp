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

main_pid="$(pgrep -fo "$MAIN_PATTERN" || true)"
if [[ -n "$main_pid" ]]; then
    echo "[wait] main experiment pid=$main_pid"
    while kill -0 "$main_pid" 2>/dev/null; do
        sleep "$POLL_SECONDS"
    done
    echo "[wait] main experiment finished"
else
    echo "[wait] no active remaining-four-categories process; scanning now"
fi

mkdir -p "$OUTPUT_ROOT"
scan_log="$OUTPUT_ROOT/retry_failed_reason_dry_run_$(date +%Y%m%d_%H%M%S).log"

set +e
"$PYTHON" -m rsr.retry_failed_reason --dry-run \
    2>&1 | tee "$scan_log"
scan_status=${PIPESTATUS[0]}
set -e

echo "[dry-run] log -> $scan_log"
exit "$scan_status"
