#!/usr/bin/env bash
#SBATCH --job-name=smartgrasp-grasp-sim
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/grasp-sim.out
#SBATCH --error=logs/grasp-sim.err

set -euo pipefail

# Slurm executes a copied script from its spool directory, so prefer the
# submission directory and fall back to the script location for local runs.
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"

WORKSPACE_DIR="${GRASPNET_WORKSPACE_DIR:-$ROOT_DIR/graspnet-workspace}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-smartgrasp}"
DEFAULT_PYTHON="/home/qiuguanhe/miniconda3/envs/${CONDA_ENV_NAME}/bin/python"

mkdir -p logs "$WORKSPACE_DIR/results"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
  conda activate "$CONDA_ENV_NAME"
else
  echo "conda command not found; falling back to PYTHON=${PYTHON:-$DEFAULT_PYTHON}" >&2
fi

if command -v proxy_status >/dev/null 2>&1; then
  PROXY_STATUS_OUTPUT="$(proxy_status 2>&1 || true)"
  echo "$PROXY_STATUS_OUTPUT"
  if ! printf '%s\n' "$PROXY_STATUS_OUTPUT" | grep -Eqi '代理.*(开启|打开|模式|已启用)|proxy.*(on|enabled|mode)|enabled|on'; then
    if command -v proxy_on >/dev/null 2>&1; then
      echo "proxy_status does not look like proxy mode; running proxy_on"
      proxy_on
      proxy_status || true
    else
      echo "proxy_on command not found; continue without changing proxy state" >&2
    fi
  fi
else
  echo "proxy_status command not found; skip proxy check" >&2
fi

PYTHON="${PYTHON:-$(command -v python 2>/dev/null || printf '%s' "$DEFAULT_PYTHON")}"
if [[ ! -x "$PYTHON" ]]; then
  echo "smartgrasp python not found or not executable: $PYTHON" >&2
  exit 1
fi

if [[ ! -d "$WORKSPACE_DIR" ]]; then
  echo "GraspNet workspace does not exist: $WORKSPACE_DIR" >&2
  exit 2
fi

ARGS=("$@")
if [[ "${ARGS[0]:-}" == "--" ]]; then
  ARGS=("${ARGS[@]:1}")
fi

has_arg() {
  local name="$1"
  local arg
  for arg in "${ARGS[@]}"; do
    if [[ "$arg" == "$name" || "$arg" == "$name="* ]]; then
      return 0
    fi
  done
  return 1
}

GRASP_OBJ_PATH="${GRASP_OBJ_PATH:-}"
GRASP_CHECKPOINT_PATH="${GRASP_CHECKPOINT_PATH:-$WORKSPACE_DIR/checkpoints/checkpoint-rs.tar}"
GRASP_TOP_K="${GRASP_TOP_K:-5}"
GRASP_DEVICE="${GRASP_DEVICE:-cuda:0}"
GRASP_OUTPUT="${GRASP_OUTPUT:-results/grasp_simulation.json}"
GRASP_RECORD_VIDEO="${GRASP_RECORD_VIDEO:-0}"
PYBULLET_GUI="${PYBULLET_GUI:-0}"
GRASP_SCALE="${GRASP_SCALE:-}"

if [[ -z "$GRASP_OBJ_PATH" ]] && ! has_arg "--obj"; then
  cat >&2 <<USAGE
Missing object mesh. Set GRASP_OBJ_PATH or pass --obj.

Examples:
  GRASP_OBJ_PATH=/path/to/textured.obj sbatch run_grasp_simulation.sh
  sbatch run_grasp_simulation.sh --obj /path/to/textured.obj --top_k 5
USAGE
  exit 2
fi

if ! has_arg "--ckpt" && [[ ! -f "$GRASP_CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found: $GRASP_CHECKPOINT_PATH" >&2
  echo "Set GRASP_CHECKPOINT_PATH or pass --ckpt /path/to/checkpoint-rs.tar" >&2
  exit 2
fi

if [[ "$GRASP_OUTPUT" = /* ]]; then
  mkdir -p "$(dirname "$GRASP_OUTPUT")"
  DISPLAY_OUTPUT="$GRASP_OUTPUT"
else
  mkdir -p "$WORKSPACE_DIR/$(dirname "$GRASP_OUTPUT")"
  DISPLAY_OUTPUT="$WORKSPACE_DIR/$GRASP_OUTPUT"
fi

export PYTHONUNBUFFERED=1
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/smartgrasp-matplotlib-${USER:-user}}"
export PYTHONPATH="$WORKSPACE_DIR:$WORKSPACE_DIR/models:$WORKSPACE_DIR/utils:$WORKSPACE_DIR/graspnet_api:${PYTHONPATH:-}"
CUDA_ROOT="${CUDA_ROOT:-$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13}"
TORCH_LIB="$("$PYTHON" - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
export CUDA_HOME="${CUDA_HOME:-$CUDA_ROOT}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$TORCH_LIB:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$MPLCONFIGDIR"

CMD=(
  "$PYTHON" -u scripts/demo_closed_loop.py
)

if [[ -n "$GRASP_OBJ_PATH" ]] && ! has_arg "--obj"; then
  CMD+=(--obj "$GRASP_OBJ_PATH")
fi

if ! has_arg "--ckpt"; then
  CMD+=(--ckpt "$GRASP_CHECKPOINT_PATH")
fi

if ! has_arg "--top_k"; then
  CMD+=(--top_k "$GRASP_TOP_K")
fi

if ! has_arg "--device"; then
  CMD+=(--device "$GRASP_DEVICE")
fi

if ! has_arg "--output"; then
  CMD+=(--output "$GRASP_OUTPUT")
fi

if [[ -n "$GRASP_SCALE" ]] && ! has_arg "--scale"; then
  CMD+=(--scale "$GRASP_SCALE")
fi

VIDEO_OUTPUT="${DISPLAY_OUTPUT%.json}_pybullet.mp4"
VIZ_OUTPUT="${DISPLAY_OUTPUT%.json}_viz_data.pkl"
CANDIDATE_PNG_OUTPUT="${DISPLAY_OUTPUT%.json}_candidates.png"
CANDIDATE_HTML_OUTPUT="${DISPLAY_OUTPUT%.json}_candidates.html"
if [[ "$GRASP_RECORD_VIDEO" == "1" || "$GRASP_RECORD_VIDEO" == "true" ]]; then
  echo "Video recording is not supported by the restored demo_closed_loop.py entrypoint." >&2
fi

if [[ "$PYBULLET_GUI" == "1" || "$PYBULLET_GUI" == "true" ]]; then
  CMD+=(--gui)
fi

CMD+=("${ARGS[@]}")

rm -f "$DISPLAY_OUTPUT" "$VIZ_OUTPUT" "$VIDEO_OUTPUT" "$CANDIDATE_PNG_OUTPUT" "$CANDIDATE_HTML_OUTPUT"

echo "Running SmartGrasp grasp simulation:"
echo "  root=$ROOT_DIR"
echo "  workspace=$WORKSPACE_DIR"
echo "  python=$PYTHON"
echo "  object=${GRASP_OBJ_PATH:-<from args>}"
echo "  checkpoint=$GRASP_CHECKPOINT_PATH"
echo "  device=$GRASP_DEVICE"
echo "  top_k=$GRASP_TOP_K"
echo "  scale=${GRASP_SCALE:-<script/default>}"
echo "  record_video=$GRASP_RECORD_VIDEO"
echo "  output=$DISPLAY_OUTPUT"
echo "  video=<disabled>"
echo "  candidate_png=<not generated by restored entrypoint>"
echo "  candidate_html=<not generated by restored entrypoint>"
echo "  gui_command=cd $WORKSPACE_DIR && python gui/app.py --results $DISPLAY_OUTPUT --viz-data ${DISPLAY_OUTPUT%.json}_viz_data.pkl --host 0.0.0.0 --port 8050"

cd "$WORKSPACE_DIR"
exec "${CMD[@]}"
