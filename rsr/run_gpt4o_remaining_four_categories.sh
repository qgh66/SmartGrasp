#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$HOME/anaconda3/envs/smartgrasp/bin/python}"
DATA_ROOT="${DATA_ROOT:-data}"
SAMPLE_INPUT_ROOT="${SAMPLE_INPUT_ROOT:-rsr/data/input}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-rsr/data/gpt4o_first10_four_categories}"
INPUT_ROOT="${INPUT_ROOT:-$EXPERIMENT_ROOT/input}"
REMAINING_INPUT_ROOT="${REMAINING_INPUT_ROOT:-$EXPERIMENT_ROOT/remaining_input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EXPERIMENT_ROOT/output}"

if [[ ! -x "$PYTHON" ]]; then
    echo "SmartGrasp Python not found: $PYTHON" >&2
    exit 1
fi

# Expand the existing first-10 input view to all 50 scene/query cases in each
# category and build a derived view containing only the remaining 40 cases.
"$PYTHON" -m rsr.prepare_gpt4o_four_categories_all \
  --data-root "$DATA_ROOT" \
  --sample-root "$SAMPLE_INPUT_ROOT" \
  --input-root "$INPUT_ROOT" \
  --remaining-input-root "$REMAINING_INPUT_ROOT"

mkdir -p "$OUTPUT_ROOT"
run_log="$OUTPUT_ROOT/run_remaining_$(date +%Y%m%d_%H%M%S).log"
overall_run_status=0

set +e
bash rsr/run_rsr.sh \
  --input-root "$REMAINING_INPUT_ROOT" \
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
  2>&1 | tee -a "$run_log"
overall_run_status=${PIPESTATUS[0]}

"$PYTHON" -m rsr.evaluate \
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
  2>&1 | tee "$OUTPUT_ROOT/evaluate_all_four_categories.log"
eval_status=${PIPESTATUS[0]}
set -e

if [[ $overall_run_status -ne 0 ]]; then
    exit "$overall_run_status"
fi
exit "$eval_status"
