#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMARTGRASP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CAMERA_SERIAL="${CAMERA_SERIAL:-243122072659}"
TOP_K="${TOP_K:-100}"
GRASP_INPUT_MODE="${GRASP_INPUT_MODE:-bbox}"
TRIAL_LOG_SUBDIR="${TRIAL_LOG_SUBDIR:-single_object}"
TRIAL_NAME="${TRIAL_NAME:-capture_only}"
VELOCITY="${VELOCITY:-10}"
ACCELERATION="${ACCELERATION:-10}"
GRASP_CROP_MARGIN_PX="${GRASP_CROP_MARGIN_PX:-50}"
GRASP_CROP_MARGIN_RATIO="${GRASP_CROP_MARGIN_RATIO:-0}"
TARGET_MASK_CENTER_TOLERANCE_PX="${TARGET_MASK_CENTER_TOLERANCE_PX:-0}"

exec "$SMARTGRASP_ROOT/run_realworld_grasp.sh" \
  --calibration-mode hand_eye \
  --hand-eye-calibration "$SMARTGRASP_ROOT/graspnet-workspace/calibration/hand_eye_tcp_camera.json" \
  --camera-serial "$CAMERA_SERIAL" \
  --top-k "$TOP_K" \
  --grasp-input-mode "$GRASP_INPUT_MODE" \
  --grasp-crop-margin-px "$GRASP_CROP_MARGIN_PX" \
  --grasp-crop-margin-ratio "$GRASP_CROP_MARGIN_RATIO" \
  --target-mask-center-tolerance-px "$TARGET_MASK_CENTER_TOLERANCE_PX" \
  --trial-log-subdir "$TRIAL_LOG_SUBDIR" \
  --trial-name "$TRIAL_NAME" \
  --velocity "$VELOCITY" \
  --acceleration "$ACCELERATION" \
  --num-cycles 1 \
  "$@"
