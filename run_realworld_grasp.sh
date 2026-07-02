#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$ROOT_DIR/graspnet-workspace"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-smartgrasp}"

cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
  conda activate "$CONDA_ENV_NAME"
else
  echo "conda command not found; continue with current Python: $(command -v python || true)" >&2
fi

if command -v proxy_status >/dev/null 2>&1; then
  PROXY_STATUS_OUTPUT="$(proxy_status 2>&1 || true)"
  echo "$PROXY_STATUS_OUTPUT"
  if ! printf '%s\n' "$PROXY_STATUS_OUTPUT" | grep -Eqi '代理.*(开启|打开|模式|已启用)|proxy.*(on|enabled|mode)|enabled|on'; then
    if command -v proxy_on >/dev/null 2>&1; then
      proxy_on
      proxy_status || true
    else
      echo "proxy_on command not found; continue without changing proxy state" >&2
    fi
  fi
else
  echo "proxy_status command not found; skip proxy check" >&2
fi

export PYTHONUNBUFFERED=1
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONPATH="$WORKSPACE_DIR:$WORKSPACE_DIR/models:$WORKSPACE_DIR/utils:$WORKSPACE_DIR/graspnet_api:${PYTHONPATH:-}"

exec python -u "$WORKSPACE_DIR/scripts/realworld_grasp.py" "$@"
