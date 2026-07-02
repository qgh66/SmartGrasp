#!/bin/bash
#SBATCH --job-name=smartgrasp_perception
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/perception_%j.out
#SBATCH --error=logs/perception_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
  source "$CONDA_HOME/etc/profile.d/conda.sh"
else
  source "$CONDA_HOME/bin/activate"
fi
conda activate smartgrasp

SCENE_ID="${SCENE_ID:-527}"
POINT_SOURCE="${POINT_SOURCE:-molmo}"
SAM_MODEL_ID="${SAM_MODEL_ID:-facebook/sam-vit-large}"
MOLMO_MODEL_ID="${MOLMO_MODEL_ID:-allenai/Molmo-7B-D-0924}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
SMARTGRASP_DATA_DIR="${SMARTGRASP_DATA_DIR:-/data/datasets/smartgrasp}"
export SMARTGRASP_DATA_DIR

COMMON_ARGS=(
  --scene-id "$SCENE_ID"
  --point-source "$POINT_SOURCE"
  --epsilon 0.01
  --kernel-size 5
  --min-contact-pixels 20
  --min-contact-ratio 0.001
)

if [ -n "$OUTPUT_ROOT" ]; then
  COMMON_ARGS+=(--output-root "$OUTPUT_ROOT")
fi

if [ "$POINT_SOURCE" = "molmo" ]; then
  COMMON_ARGS+=(
    --molmo-model-id "$MOLMO_MODEL_ID"
    --molmo-device cuda
    --sam-model-id "$SAM_MODEL_ID"
    --sam-point-grid-radius 10
    --mask-clean-kernel 3
    --device cuda
  )
fi

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Repo: $PWD"
echo "Python: $(which python)"
python -V
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SMARTGRASP_DATA_DIR=$SMARTGRASP_DATA_DIR"
echo "Running scene_id=$SCENE_ID point_source=$POINT_SOURCE"

python -u perception/perception.py "${COMMON_ARGS[@]}"
