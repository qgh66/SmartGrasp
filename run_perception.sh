#!/usr/bin/env bash
#SBATCH --job-name=smartgrasp-perception
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/perception-%j.out
#SBATCH --error=logs/perception-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"

mkdir -p logs

export SMARTGRASP_DATA_DIR="${SMARTGRASP_DATA_DIR:-/home/data/datasets/FreeGraspData}"
export HF_HOME="/home/data/models/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="/home/data/models/torch"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/smartgrasp-matplotlib-${USER:-user}}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON="${PYTHON:-/home/qiuguanhe/miniconda3/envs/smartgrasp/bin/python}"
SEGMENTATION_BACKEND="${SEGMENTATION_BACKEND:-sam2-molmo-langsam}"
POINT_SOURCE="${POINT_SOURCE:-molmo}"

if [[ ! -x "$PYTHON" ]]; then
  echo "smartgrasp python not found or not executable: $PYTHON" >&2
  exit 1
fi

for required_dir in "$SMARTGRASP_DATA_DIR" "$(dirname "$HF_HOME")"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "Required directory does not exist: $required_dir" >&2
    echo "Please create it with suitable permissions before running this Slurm job." >&2
    exit 2
  fi
done

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" "$MPLCONFIGDIR"

if ! compgen -G "$SMARTGRASP_DATA_DIR/*.parquet" > /dev/null; then
  echo "No parquet files found in SMARTGRASP_DATA_DIR=$SMARTGRASP_DATA_DIR" >&2
  echo "Download or place the FreeGrasp parquet files and npz_file.zip under /home/data/datasets/FreeGraspData first." >&2
  exit 2
fi

if [[ ! -e "$SMARTGRASP_DATA_DIR/npz_file.zip" ]] && ! find "$SMARTGRASP_DATA_DIR" -name '*.npz' -print -quit | grep -q .; then
  echo "No npz_file.zip or .npz files found in SMARTGRASP_DATA_DIR=$SMARTGRASP_DATA_DIR" >&2
  exit 2
fi

if [[ "$SEGMENTATION_BACKEND" == "sam2-molmo-langsam" && "$POINT_SOURCE" != "molmo" ]]; then
  echo "sam2-molmo-langsam requires POINT_SOURCE=molmo, got POINT_SOURCE=$POINT_SOURCE" >&2
  exit 2
fi

echo "Running SmartGrasp perception:"
echo "  scene_id=${SCENE_ID:-527}"
echo "  point_source=$POINT_SOURCE"
echo "  segmentation_backend=$SEGMENTATION_BACKEND"
echo "  python=$PYTHON"

EXTRA_ARGS=()
if [[ -n "${SAVE_CANDIDATES:-}" ]]; then
  EXTRA_ARGS+=(--save-candidates)
fi

"$PYTHON" -u perception/perception.py \
  --scene-id "${SCENE_ID:-527}" \
  --point-source "$POINT_SOURCE" \
  --molmo-model-id "${MOLMO_MODEL_ID:-allenai/Molmo-7B-D-0924}" \
  --review-model-id "${REVIEW_MODEL_ID:-gpt-5.5}" \
  --review-api-key-env "${REVIEW_API_KEY_ENV:-OPENAI_API_KEY}" \
  --review-base-url "${REVIEW_BASE_URL:-${OPENAI_BASE_URL:-}}" \
  --review-timeout "${REVIEW_TIMEOUT:-120}" \
  --segmentation-backend "$SEGMENTATION_BACKEND" \
  --sam-model-id "${SAM_MODEL_ID:-facebook/sam-vit-large}" \
  --epsilon "${EPSILON:-0.05}" \
  --kernel-size "${KERNEL_SIZE:-5}" \
  --min-contact-pixels "${MIN_CONTACT_PIXELS:-50}" \
  --min-contact-ratio "${MIN_CONTACT_RATIO:-0.002}" \
  --sam-point-grid-radius "${SAM_POINT_GRID_RADIUS:-10}" \
  --sam-prompt-mode "${SAM_PROMPT_MODE:-grid}" \
  --sam-negative-points "${SAM_NEGATIVE_POINTS:-0}" \
  --mask-clean-kernel "${MASK_CLEAN_KERNEL:-3}" \
  --proposal-backend "${PROPOSAL_BACKEND:-sam2-auto}" \
  --proposal-min-area-ratio "${PROPOSAL_MIN_AREA_RATIO:-0.0015}" \
  --proposal-max-area-ratio "${PROPOSAL_MAX_AREA_RATIO:-0.11}" \
  --proposal-iou-threshold "${PROPOSAL_IOU_THRESHOLD:-0.35}" \
  --proposal-containment-threshold "${PROPOSAL_CONTAINMENT_THRESHOLD:-0.6}" \
  --proposal-border-fraction-threshold "${PROPOSAL_BORDER_FRACTION_THRESHOLD:-0.18}" \
  --max-proposal-masks "${MAX_PROPOSAL_MASKS:-3}" \
  --depth-merge-threshold "${DEPTH_MERGE_THRESHOLD:-0.0}" \
  --anchor-merge-depth-threshold "${ANCHOR_MERGE_DEPTH_THRESHOLD:-0.015}" \
  --sam2-points-per-side "${SAM2_POINTS_PER_SIDE:-32}" \
  --sam2-crop-n-layers "${SAM2_CROP_N_LAYERS:-0}" \
  --sam2-pred-iou-thresh "${SAM2_PRED_IOU_THRESH:-0.7}" \
  --sam2-stability-score-thresh "${SAM2_STABILITY_SCORE_THRESH:-0.88}" \
  --preserve-unclaimed-sam2 "${PRESERVE_UNCLAIMED_SAM2:-24}" \
  "${EXTRA_ARGS[@]}" \
  --device "${DEVICE:-cuda}" \
  "$@"
