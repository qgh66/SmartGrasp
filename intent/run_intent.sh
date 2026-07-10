#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-/home/qiuguanhe/miniconda3/envs/smartgrasp/bin/python}"

MODE="${1:-single}"
shift || true

export_scene() {
  local scene_id="$1"
  local scene_dir="$ROOT_DIR/data/scene_${scene_id}"

  if [[ -n "${EXPORT_LOCAL_DIR:-}" ]]; then
    mkdir -p "$EXPORT_LOCAL_DIR"
    cp -a "$scene_dir" "$EXPORT_LOCAL_DIR/"
    echo "[scene=$scene_id] exported to local dir: $EXPORT_LOCAL_DIR/scene_${scene_id}"
  fi

  if [[ -n "${EXPORT_SCP_TARGET:-}" ]]; then
    scp -r "$scene_dir" "$EXPORT_SCP_TARGET"
    echo "[scene=$scene_id] exported via scp to: $EXPORT_SCP_TARGET"
  fi
}

case "$MODE" in
  single)
    export SCENE_ID="${SCENE_ID:-1094}"
    export POINT_SOURCE="${POINT_SOURCE:-molmo}"
    export SEGMENTATION_BACKEND="${SEGMENTATION_BACKEND:-sam2-molmo-langsam}"
    bash perception/run_perception.sh "$@"
    export_scene "$SCENE_ID"
    ;;

  single-scp)
    export SCENE_ID="${SCENE_ID:-1094}"
    export POINT_SOURCE="${POINT_SOURCE:-molmo}"
    export SEGMENTATION_BACKEND="${SEGMENTATION_BACKEND:-sam2-molmo-langsam}"
    bash perception/run_perception.sh "$@"
    export_scene "$SCENE_ID"
    ;;

  batch)
    SCENE_LIST_FILE="${SCENE_LIST_FILE:-}"
    COUNT="${COUNT:-500}"
    BATCH_ORDER="${BATCH_ORDER:-first}"

    if [[ -n "$SCENE_LIST_FILE" ]]; then
      if [[ ! -f "$SCENE_LIST_FILE" ]]; then
        echo "Scene list file not found: $SCENE_LIST_FILE" >&2
        exit 2
      fi
      mapfile -t ALL_SCENES < <(grep -E '^[0-9]+$' "$SCENE_LIST_FILE" | sort -n | awk '!seen[$0]++')
      if [[ "${#ALL_SCENES[@]}" -eq 0 ]]; then
        echo "No valid scene ids found in $SCENE_LIST_FILE" >&2
        exit 2
      fi
    else
      mapfile -t ALL_SCENES < <("$PYTHON" - <<'PY'
import glob
import os
import pandas as pd
from pathlib import Path

data_dir = Path(os.environ.get("SMARTGRASP_DATA_DIR", "/home/data/datasets/FreeGraspData"))
parquet_files = sorted(glob.glob(str(data_dir / "*.parquet")))
if not parquet_files:
    raise SystemExit("No parquet files found under SMARTGRASP_DATA_DIR")

frames = [pd.read_parquet(path, columns=["sceneId"]) for path in parquet_files]
df = pd.concat(frames, ignore_index=True)
scene_ids = sorted({int(x) for x in df["sceneId"].dropna().tolist()})
for scene_id in scene_ids:
    print(scene_id)
PY
)
      if [[ "${#ALL_SCENES[@]}" -eq 0 ]]; then
        echo "No valid scene ids found from SMARTGRASP_DATA_DIR=$SMARTGRASP_DATA_DIR" >&2
        exit 2
      fi
    fi

    if (( COUNT > ${#ALL_SCENES[@]} )); then
      COUNT="${#ALL_SCENES[@]}"
    fi

    case "$BATCH_ORDER" in
      first)
        mapfile -t SELECTED_SCENES < <(printf '%s\n' "${ALL_SCENES[@]}" | head -n "$COUNT")
        ;;
      random)
        mapfile -t SELECTED_SCENES < <(printf '%s\n' "${ALL_SCENES[@]}" | shuf -n "$COUNT" | sort -n)
        ;;
      *)
        echo "Unsupported BATCH_ORDER: $BATCH_ORDER (use first or random)" >&2
        exit 2
        ;;
    esac

    if [[ -n "$SCENE_LIST_FILE" ]]; then
      echo "Selected ${#SELECTED_SCENES[@]} scenes from $SCENE_LIST_FILE (order=$BATCH_ORDER)"
    else
      echo "Selected ${#SELECTED_SCENES[@]} scenes from SMARTGRASP_DATA_DIR=$SMARTGRASP_DATA_DIR (order=$BATCH_ORDER)"
    fi
    printf '%s\n' "${SELECTED_SCENES[@]}" | tee selected_scenes.txt >/dev/null

    for scene_id in "${SELECTED_SCENES[@]}"; do
      echo "[scene=$scene_id] generating perception"
      SCENE_ID="$scene_id" \
      POINT_SOURCE="${POINT_SOURCE:-molmo}" \
      SEGMENTATION_BACKEND="${SEGMENTATION_BACKEND:-sam2-molmo-langsam}" \
      bash perception/run_perception.sh
      export_scene "$scene_id"
    done
    ;;

  range)
    SCENE_START="${SCENE_START:-}"
    SCENE_END="${SCENE_END:-}"

    if [[ -z "$SCENE_START" || -z "$SCENE_END" ]]; then
      echo "SCENE_START and SCENE_END are required in range mode." >&2
      exit 2
    fi
    if (( SCENE_START > SCENE_END )); then
      echo "SCENE_START must be <= SCENE_END." >&2
      exit 2
    fi

    for scene_id in $(seq "$SCENE_START" "$SCENE_END"); do
      echo "[scene=$scene_id] generating perception"
      SCENE_ID="$scene_id" \
      POINT_SOURCE="${POINT_SOURCE:-molmo}" \
      SEGMENTATION_BACKEND="${SEGMENTATION_BACKEND:-sam2-molmo-langsam}" \
      bash perception/run_perception.sh
      export_scene "$scene_id"
    done
    ;;

  *)
    echo "Usage:"
    echo "  bash intent/run_intent.sh single"
    echo "  bash intent/run_intent.sh single-scp"
    echo "  bash intent/run_intent.sh batch"
    echo "  bash intent/run_intent.sh range"
    echo ""
    echo "single mode env:"
    echo "  SCENE_ID, POINT_SOURCE, SEGMENTATION_BACKEND"
    echo "  EXPORT_LOCAL_DIR or EXPORT_SCP_TARGET (optional)"
    echo "  example: EXPORT_SCP_TARGET=user@notebook:/path/to/save/"
    echo ""
    echo "range mode env:"
    echo "  SCENE_START, SCENE_END, POINT_SOURCE, SEGMENTATION_BACKEND"
    echo "  EXPORT_LOCAL_DIR or EXPORT_SCP_TARGET (optional)"
    echo "  example: EXPORT_SCP_TARGET=user@notebook:/path/to/save/"
    echo ""
    echo "batch mode env:"
    echo "  SCENE_LIST_FILE(optional), COUNT, BATCH_ORDER, POINT_SOURCE, SEGMENTATION_BACKEND"
    echo "  EXPORT_LOCAL_DIR or EXPORT_SCP_TARGET (optional)"
    echo "  example: EXPORT_SCP_TARGET=user@notebook:/path/to/save/"
    exit 2
    ;;
esac
