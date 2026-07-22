#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$HOME/anaconda3/envs/smartgrasp/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-rsr/data/hard_ambi_all/input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-rsr/data/hard_ambi_all/output}"

if [[ ! -f "$INPUT_ROOT/manifest.json" ]]; then
    "$PYTHON" -m rsr.prepare_hard_ambi_all --input-root "$INPUT_ROOT"
fi

set +e
bash rsr/run_rsr.sh \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --testcase hard_ambiguous \
  --reason-model gpt-5.5 \
  --algorithm information_gain_original \
  --algorithm information_gain_graspability \
  --algorithm theory_original \
  --algorithm theory_graspability \
  --perception-mode vlm \
  --perception-review-model gpt-5.5 \
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
  --force-perception \
  --force-reason
run_status=$?

"$PYTHON" -m rsr.evaluate \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --model gpt-5.5 \
  --algorithm information_gain_original \
  --algorithm information_gain_graspability \
  --algorithm theory_original \
  --algorithm theory_graspability
eval_status=$?
set -e

if [[ $run_status -ne 0 ]]; then
    exit "$run_status"
fi
exit "$eval_status"
