#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-smartgrasp}"
DEFAULT_PYTHON="/home/admin128/anaconda3/envs/${CONDA_ENV_NAME}/bin/python"

cd "$ROOT_DIR"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/smartgrasp-matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]]; then
  set +u
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
  elif [[ -f /home/admin128/anaconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck source=/dev/null
    source /home/admin128/anaconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV_NAME"
  fi
  set -u
fi

# Conda activation can replace PYTHONPATH, so add the workspace afterwards.
export PYTHONPATH="$ROOT_DIR/graspnet-workspace:${PYTHONPATH:-}"

if command -v proxy_status >/dev/null 2>&1; then
  PROXY_STATUS_OUTPUT="$(proxy_status 2>&1 || true)"
  echo "$PROXY_STATUS_OUTPUT"
  if ! printf '%s\n' "$PROXY_STATUS_OUTPUT" | grep -Eqi \
      '代理.*(开启|打开|模式|已启用)|proxy.*(on|enabled|mode)|enabled|on'; then
    if command -v proxy_on >/dev/null 2>&1; then
      proxy_on
      proxy_status || true
    else
      echo "proxy_on command not found; cannot enable proxy" >&2
      exit 1
    fi
  fi
elif [[ -n "${HTTPS_PROXY:-${https_proxy:-}}" ]]; then
  echo "proxy_status command not found; HTTPS proxy environment is configured"
else
  echo "proxy_status command not found and no HTTPS proxy is configured" >&2
  exit 1
fi

PYTHON_EXECUTABLE="$(command -v python 2>/dev/null || true)"
if [[ -z "$PYTHON_EXECUTABLE" ]] && [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON_EXECUTABLE="$DEFAULT_PYTHON"
fi
if [[ -z "$PYTHON_EXECUTABLE" ]] || [[ ! -x "$PYTHON_EXECUTABLE" ]]; then
  echo "Python executable for smartgrasp was not found" >&2
  exit 1
fi

exec "$PYTHON_EXECUTABLE" -m simulation.capture_artifacts "$@"
