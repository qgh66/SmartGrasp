#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMARTGRASP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CAMERA_SERIAL="${CAMERA_SERIAL:-76630}"
TOP_K="${TOP_K:-100}"
GRASP_INPUT_MODE="${GRASP_INPUT_MODE:-bbox}"
CANDIDATE_INDEX="${CANDIDATE_INDEX:-0}"
TRIAL_LOG_SUBDIR="${TRIAL_LOG_SUBDIR:-single_object}"
TRIAL_NAME="${TRIAL_NAME:-grasp_execute_once}"
VELOCITY="${VELOCITY:-10}"
ACCELERATION="${ACCELERATION:-10}"
APPROACH_OFFSET_MM="${APPROACH_OFFSET_MM:-100}"
LIFT_MM="${LIFT_MM:-80}"
PLACE_RELEASE_LOWER_MM="${PLACE_RELEASE_LOWER_MM:-100}"
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
  --candidate-index "$CANDIDATE_INDEX" \
  --velocity "$VELOCITY" \
  --acceleration "$ACCELERATION" \
  --approach-offset-mm "$APPROACH_OFFSET_MM" \
  --lift-mm "$LIFT_MM" \
  --place-release-lower-mm "$PLACE_RELEASE_LOWER_MM" \
  --num-cycles 1 \
  --execute \
  "$@"
