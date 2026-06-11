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
export OPENAI_API_KEY='sk-0icgaaSMWa6ZBEmzKE960dC35DPmPuzUzN7hTGuFofOUCcHm'
export OPENAI_BASE_URL=https://www.highland-api.top/v1

PYTHON="${PYTHON:-/home/qiuguanhe/miniconda3/envs/smartgrasp/bin/python}"
SEGMENTATION_BACKEND="${SEGMENTATION_BACKEND:-sam2-langsam}"

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

echo "Running SmartGrasp perception:"
echo "  scene_id=${SCENE_ID:-527}"
echo "  segmentation_backend=$SEGMENTATION_BACKEND"
echo "  python=$PYTHON"

EXTRA_ARGS=()
if [[ -n "${SAVE_CANDIDATES:-}" ]]; then
  EXTRA_ARGS+=(--save-candidates)
fi

"$PYTHON" -u perception/perception.py \
  --scene-id "${SCENE_ID:-527}" \
  --review-model-id "${REVIEW_MODEL_ID:-gpt-5.5}" \
  --review-api-key-env "${REVIEW_API_KEY_ENV:-OPENAI_API_KEY}" \
  --review-base-url "${REVIEW_BASE_URL:-${OPENAI_BASE_URL:-}}" \
  --review-timeout "${REVIEW_TIMEOUT:-120}" \
  --epsilon "${EPSILON:-0.05}" \
  --kernel-size "${KERNEL_SIZE:-5}" \
  --min-contact-pixels "${MIN_CONTACT_PIXELS:-50}" \
  --min-contact-ratio "${MIN_CONTACT_RATIO:-0.002}" \
  --mask-clean-kernel "${MASK_CLEAN_KERNEL:-3}" \
  --proposal-min-area-ratio "${PROPOSAL_MIN_AREA_RATIO:-0.006}" \
  --proposal-max-area-ratio "${PROPOSAL_MAX_AREA_RATIO:-0.11}" \
  --proposal-border-fraction-threshold "${PROPOSAL_BORDER_FRACTION_THRESHOLD:-0.18}" \
  --sam2-points-per-side "${SAM2_POINTS_PER_SIDE:-32}" \
  --sam2-crop-n-layers "${SAM2_CROP_N_LAYERS:-0}" \
  --sam2-pred-iou-thresh "${SAM2_PRED_IOU_THRESH:-0.7}" \
  --sam2-stability-score-thresh "${SAM2_STABILITY_SCORE_THRESH:-0.88}" \
  --preserve-unclaimed-sam2 "${PRESERVE_UNCLAIMED_SAM2:-24}" \
  "${EXTRA_ARGS[@]}" \
  --device "${DEVICE:-cuda}" \
  "$@"
