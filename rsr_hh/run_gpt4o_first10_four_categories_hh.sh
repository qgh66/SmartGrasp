#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$HOME/anaconda3/envs/smartgrasp/bin/python}"
SOURCE_INPUT_ROOT="${SOURCE_INPUT_ROOT:-rsr_hh/data/input}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-rsr_hh/data/gpt4o_first10_four_categories}"
INPUT_ROOT="${INPUT_ROOT:-$EXPERIMENT_ROOT/input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EXPERIMENT_ROOT/output}"

if [[ ! -x "$PYTHON" ]]; then
    echo "SmartGrasp Python not found: $PYTHON" >&2
    exit 1
fi

"$PYTHON" -m rsr_hh.prepare_gpt4o_first10 \
  --source-root "$SOURCE_INPUT_ROOT" \
  --target-root "$INPUT_ROOT" \
  --limit 10

mkdir -p "$OUTPUT_ROOT"
run_log="$OUTPUT_ROOT/run_$(date +%Y%m%d_%H%M%S).log"

force_args=()
[[ "${FORCE_PERCEPTION:-0}" == "1" ]] && force_args+=(--force-perception)
[[ "${FORCE_REASON:-0}" == "1" ]] && force_args+=(--force-reason)

set +e
bash rsr_hh/run_rsr_hh.sh \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --testcase hard_ambiguous \
  --testcase medium_ambiguous \
  --testcase medium_unambiguous \
  --testcase hard_unambiguous \
  --reason-model gpt-4o \
  --algorithm information_gain_original \
  --algorithm information_gain_graspability \
  --algorithm theory_original \
  --algorithm theory_graspability \
  --perception-mode vlm \
  --perception-review-model gpt-4o \
  --perception-review-timeout 300 \
  --sam2-points-per-side 24 \
  --sam2-pred-iou-thresh 0.68 \
  --sam2-stability-score-thresh 0.83 \
  --sam2-crop-n-layers 0 \
  --depth-sam2-crop-n-layers 1 \
  --depth-sam2-pred-iou-thresh 0.58 \
  --depth-sam2-stability-score-thresh 0.73 \
  --kernel-size 11 \
  --min-contact-pixels 50 \
  --min-contact-ratio 0.002 \
  --mask-clean-kernel 3 \
  --proposal-min-area-ratio 0.006 \
  --proposal-max-area-ratio 0.11 \
  --proposal-border-fraction-threshold 0.18 \
  "${force_args[@]}" \
  2>&1 | tee "$run_log"
run_status=${PIPESTATUS[0]}

"$PYTHON" -m rsr_hh.evaluate \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --testcase hard_ambiguous \
  --testcase medium_ambiguous \
  --testcase medium_unambiguous \
  --testcase hard_unambiguous \
  --model gpt-4o \
  --algorithm information_gain_original \
  --algorithm information_gain_graspability \
  --algorithm theory_original \
  --algorithm theory_graspability \
  2>&1 | tee "$OUTPUT_ROOT/evaluate.log"
eval_status=${PIPESTATUS[0]}
set -e

if [[ $run_status -ne 0 ]]; then
    exit "$run_status"
fi
exit "$eval_status"
