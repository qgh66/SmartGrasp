#!/usr/bin/env bash
#SBATCH --job-name=smartgrasp-freegrasp-style
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/freegrasp-style-%j.out
#SBATCH --error=logs/freegrasp-style-%j.err

set -uo pipefail

BENCHMARK_ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BENCHMARK_ROOT_DIR"

BENCHMARK_CONDA_ENV="${CONDA_ENV_NAME:-smartgrasp}"
set +u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "/home/admin128/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source /home/admin128/anaconda3/etc/profile.d/conda.sh
fi
conda activate "$BENCHMARK_CONDA_ENV"
set -u

if command -v proxy_status >/dev/null 2>&1; then
  BENCHMARK_PROXY_STATUS="$(proxy_status 2>&1 || true)"
  echo "$BENCHMARK_PROXY_STATUS"
  if ! printf '%s\n' "$BENCHMARK_PROXY_STATUS" | grep -Eqi '代理.*(开启|打开|模式|已启用)|proxy.*(on|enabled|mode)|enabled|on'; then
    if command -v proxy_on >/dev/null 2>&1; then
      proxy_on
      proxy_status || true
    else
      echo "proxy_on command not found; keep the inherited proxy environment" >&2
    fi
  fi
else
  echo "proxy_status command not found; keep the inherited proxy environment" >&2
fi

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://yunwu.ai/v1}"
export REVIEW_MODEL_ID="${REVIEW_MODEL_ID:-gpt-5.5}"
export REASON_MODEL="${REASON_MODEL:-gpt-5.5}"
export REASON_PRIOR_PROMPT="${REASON_PRIOR_PROMPT:-graspability}"
export REASON_RANKING_SCORE="${REASON_RANKING_SCORE:-ig_graspability}"
export PYBULLET_GUI=0
export GRASP_RECORD_VIDEO=0
export GRASP_GUI_SPEED=1.0

BENCHMARK_MASK_MIN_IOU="${FREEGRASP_MASK_MIN_IOU:-0.5}"
BENCHMARK_MAX_TASK_ROUNDS="${FREEGRASP_MAX_TASK_ROUNDS:-1}"
BENCHMARK_RUN_ID="${FREEGRASP_RUN_ID:-job_${SLURM_JOB_ID:-manual}}"
BENCHMARK_RESULT_DIR="$BENCHMARK_ROOT_DIR/graspnet-workspace/results/freegrasp_style/$BENCHMARK_RUN_ID"
BENCHMARK_MANIFEST_PATH="$BENCHMARK_RESULT_DIR/run_manifest.tsv"
BENCHMARK_FAILURE_LOG="$BENCHMARK_RESULT_DIR/failed_runs.txt"
BENCHMARK_REPORT_PATH="$BENCHMARK_ROOT_DIR/FREEGRASP_STYLE_RESULTS.md"
BENCHMARK_PYTHON="${CONDA_PREFIX}/bin/python"

mkdir -p "$BENCHMARK_ROOT_DIR/logs" "$BENCHMARK_RESULT_DIR"
printf 'condition\tepisode\trepeat\tprompt_template\tscene_config\ttarget_names\tresult_path\n' > "$BENCHMARK_MANIFEST_PATH"
: > "$BENCHMARK_FAILURE_LOG"

BENCHMARK_PROMPTS=(
  '抓取{target}'
  '请找到{target}并将它取出'
  '目标物体是{target}，请将它抓取出来'
)

BENCHMARK_FAILURE_COUNT=0

run_benchmark_episode() {
  local condition_key="$1"
  local scene_config="$2"
  local episode_name="$3"
  shift 3
  local target_names=("$@")
  local target_names_csv
  local prompt_index
  local repeat
  local prompt_template
  local result_path
  local result_name
  local exit_code

  printf -v target_names_csv '%s,' "${target_names[@]}"
  target_names_csv="${target_names_csv%,}"

  for prompt_index in "${!BENCHMARK_PROMPTS[@]}"; do
    repeat=$((prompt_index + 1))
    prompt_template="${BENCHMARK_PROMPTS[$prompt_index]}"
    result_name="${condition_key}__${episode_name}__r${repeat}.json"
    result_path="$BENCHMARK_RESULT_DIR/$result_name"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$condition_key" \
      "$episode_name" \
      "$repeat" \
      "$prompt_template" \
      "$scene_config" \
      "$target_names_csv" \
      "$result_path" >> "$BENCHMARK_MANIFEST_PATH"

    echo
    echo "================================================================"
    echo "condition=$condition_key episode=$episode_name repeat=$repeat"
    echo "prompt=$prompt_template"
    echo "targets=$target_names_csv"
    echo "result=$result_path"
    echo "================================================================"

    if bash "$BENCHMARK_ROOT_DIR/run_grasp_simulation.sh" \
      --scene-config "$scene_config" \
      --instruction "$prompt_template" \
      --run-pipeline-after-capture \
      --perception-reason-test \
      --continuous-grasp \
      --skip-viz-data \
      --target-objects "${target_names[@]}" \
      --max-task-rounds "$BENCHMARK_MAX_TASK_ROUNDS" \
      --max-stalled-passes 1 \
      --target-mask-min-iou "$BENCHMARK_MASK_MIN_IOU" \
      --reobserve-settle-steps 0 \
      --initial-pose-hold-seconds 0 \
      --seed "$repeat" \
      --output "$result_path"; then
      echo "completed: $result_name"
    else
      exit_code=$?
      BENCHMARK_FAILURE_COUNT=$((BENCHMARK_FAILURE_COUNT + 1))
      printf '%s\texit_code=%s\n' "$result_name" "$exit_code" >> "$BENCHMARK_FAILURE_LOG"
      echo "failed: $result_name (exit_code=$exit_code)" >&2
    fi
  done
}

FLAT_SCENE="$BENCHMARK_ROOT_DIR/graspnet-workspace/config/industrial_scene.json"
STACKED_SCENE="$BENCHMARK_ROOT_DIR/graspnet-workspace/config/industrial_scene_stacked.json"

echo "Starting SmartGrasp FreeGrasp-style benchmark"
echo "run_id=$BENCHMARK_RUN_ID"
echo "conda_env=$CONDA_DEFAULT_ENV"
echo "headless=1"
echo "graspnet=disabled"
echo "mask_min_iou=$BENCHMARK_MASK_MIN_IOU"
echo "max_task_rounds=$BENCHMARK_MAX_TASK_ROUNDS"

echo "[1/6] Low complexity, without ambiguity"
for BENCHMARK_TARGET in adjustable_wrench power_drill battery; do
  run_benchmark_episode \
    low_without_ambiguity "$FLAT_SCENE" "$BENCHMARK_TARGET" "$BENCHMARK_TARGET"
done

echo "[2/6] Low complexity, with ambiguity"
for BENCHMARK_TARGET in small_clamp medium_clamp large_clamp; do
  run_benchmark_episode \
    low_with_ambiguity "$FLAT_SCENE" "$BENCHMARK_TARGET" "$BENCHMARK_TARGET"
done

echo "[3/6] Medium complexity, without ambiguity"
for BENCHMARK_TARGET in power_drill_cover_a adjustable_wrench_cover_b two_color_hammer_cover_d; do
  run_benchmark_episode \
    medium_without_ambiguity "$STACKED_SCENE" "$BENCHMARK_TARGET" "$BENCHMARK_TARGET"
done

echo "[4/6] Medium complexity, with ambiguity"
for BENCHMARK_TARGET in flat_screwdriver_cover_c battery_cover_e medium_clamp_cover_f; do
  run_benchmark_episode \
    medium_with_ambiguity "$STACKED_SCENE" "$BENCHMARK_TARGET" "$BENCHMARK_TARGET"
done

echo "[5/6] High complexity, without ambiguity"
run_benchmark_episode \
  high_without_ambiguity \
  "$STACKED_SCENE" \
  hammer_then_phillips \
  two_color_hammer_cover_d \
  phillips_screwdriver_base_d

echo "[6/6] High complexity, with ambiguity"
run_benchmark_episode \
  high_with_ambiguity \
  "$STACKED_SCENE" \
  drill_then_battery \
  power_drill_cover_a \
  battery_fully_occluded_a

if ! "$BENCHMARK_PYTHON" \
  "$BENCHMARK_ROOT_DIR/graspnet-workspace/scripts/summarize_freegrasp_style_results.py" \
  --manifest "$BENCHMARK_MANIFEST_PATH" \
  --failure-log "$BENCHMARK_FAILURE_LOG" \
  --run-id "$BENCHMARK_RUN_ID" \
  --mask-min-iou "$BENCHMARK_MASK_MIN_IOU" \
  --output "$BENCHMARK_REPORT_PATH"; then
  echo "Failed to generate benchmark Markdown report" >&2
  exit 2
fi

echo
echo "Benchmark finished"
echo "result_dir=$BENCHMARK_RESULT_DIR"
echo "report=$BENCHMARK_REPORT_PATH"
echo "failed_runs=$BENCHMARK_FAILURE_COUNT"

if ((BENCHMARK_FAILURE_COUNT > 0)); then
  exit 1
fi
