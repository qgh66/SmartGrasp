#!/usr/bin/env bash
#SBATCH --job-name=reason_compare
#SBATCH --output=logs/reason_compare_%j.out
#SBATCH --error=logs/reason_compare_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

USE_PROXY="${USE_PROXY:-0}"
if [[ "$USE_PROXY" != "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

PYTHON_BIN="${PYTHON_BIN:-/home/qiuguanhe/miniconda3/envs/smartgrasp/bin/python}"
DATA_ROOT="${DATA_ROOT:-data}"
LIMIT="${LIMIT:-10}"
OUT_ROOT="${OUT_ROOT:-runs_reason_compare}"
PRIOR_PROMPT="${PRIOR_PROMPT:-original}"
MODELS="${MODELS:-gpt-5.5}"
ALGORITHMS="${ALGORITHMS:-legacy ig ig_graspability theory}"

read -r -a MODEL_LIST <<< "$MODELS"
read -r -a ALGORITHM_LIST <<< "$ALGORITHMS"

echo "[CONFIG] python        = $PYTHON_BIN"
echo "[CONFIG] data root     = $DATA_ROOT"
echo "[CONFIG] limit         = $LIMIT"
echo "[CONFIG] output root   = $OUT_ROOT"
echo "[CONFIG] prior prompt  = $PRIOR_PROMPT"
echo "[CONFIG] models        = ${MODEL_LIST[*]}"
echo "[CONFIG] algorithms    = ${ALGORITHM_LIST[*]}"
echo "[CONFIG] use proxy     = $USE_PROXY"

for model in "${MODEL_LIST[@]}"; do
  for algorithm in "${ALGORITHM_LIST[@]}"; do
    echo
    echo "===== run reason: model=$model algorithm=$algorithm ====="
    "$PYTHON_BIN" -m reason.run_reason \
      --root "$DATA_ROOT" \
      --limit "$LIMIT" \
      --closed-loop \
      --model "$model" \
      --prior-prompt "$PRIOR_PROMPT" \
      --reason-algorithm "$algorithm" \
      --out-root "$OUT_ROOT"
  done
done

echo
echo "===== analyze reason comparison ====="
"$PYTHON_BIN" analyze_reason_experiment.py --root "$OUT_ROOT"

echo
echo "[DONE] closed-loop table:"
cat "$OUT_ROOT/analysis/closed_loop_summary.md"

echo
echo "[DONE] reason parameter table:"
cat "$OUT_ROOT/analysis/reason_param_summary.md"
