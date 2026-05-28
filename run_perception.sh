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
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-/home/qiuguanhe/miniconda3/envs/smartgrasp/bin/python}"

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

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME"

if ! compgen -G "$SMARTGRASP_DATA_DIR/*.parquet" > /dev/null; then
  echo "No parquet files found in SMARTGRASP_DATA_DIR=$SMARTGRASP_DATA_DIR" >&2
  echo "Download or place the FreeGrasp parquet files and npz_file.zip under /home/data/datasets/FreeGraspData first." >&2
  exit 2
fi

if [[ ! -e "$SMARTGRASP_DATA_DIR/npz_file.zip" ]] && ! find "$SMARTGRASP_DATA_DIR" -name '*.npz' -print -quit | grep -q .; then
  echo "No npz_file.zip or .npz files found in SMARTGRASP_DATA_DIR=$SMARTGRASP_DATA_DIR" >&2
  exit 2
fi

"$PYTHON" -u perception/perception.py \
  --scene-id "${SCENE_ID:-527}" \
  --point-source "${POINT_SOURCE:-molmo}" \
  --molmo-model-id "${MOLMO_MODEL_ID:-allenai/Molmo-7B-D-0924}" \
  --segmentation-backend "${SEGMENTATION_BACKEND:-langsam}" \
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
  --device "${DEVICE:-cuda}" \
  "$@"
