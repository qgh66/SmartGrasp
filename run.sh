#!/bin/bash
#SBATCH --job-name=smartgrasp-perception
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run-%j.out
#SBATCH --error=logs/run-%j.err

#SCENE_ID=823 sbatch run.sh
set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"

mkdir -p logs

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

source /home/qiuguanhe/miniconda3/etc/profile.d/conda.sh
conda activate smartgrasp

export SCENE_ID="${SCENE_ID:-184}"
export DEVICE="${DEVICE:-cuda}"

./run_perception.sh "$@"
