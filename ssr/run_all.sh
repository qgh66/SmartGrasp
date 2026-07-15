#!/usr/bin/env bash
# 逐 scene 完整处理：perception → gt → intent ×3 → reason ×3 → 放入分类目录
# 用法: bash ssr/run_all.sh [category] [scene_ids ...]

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ======== Perception 参数 ========
PERCEPTION_ARGS="
  --sam2-points-per-side 24
  --sam2-pred-iou-thresh 0.68
  --sam2-stability-score-thresh 0.83
  --sam2-crop-n-layers 0
  --depth-sam2-crop-n-layers 1
  --depth-sam2-pred-iou-thresh 0.58
  --depth-sam2-stability-score-thresh 0.73
  --kernel-size 11
  --min-contact-pixels 50
  --min-contact-ratio 0.002
  --mask-clean-kernel 3
  --proposal-min-area-ratio 0.006
  --proposal-max-area-ratio 0.11
  --proposal-border-fraction-threshold 0.18
  --review-model-id gpt-5.5
  --review-timeout 300
"
PERCEPTION_ARGS_FLAT=$(echo $PERCEPTION_ARGS)

# ======== Reason 参数 ========
REASON_MODEL="gpt-4o"
INTENT_MODEL="gpt-4o"
REASON_ARGS="
  --model ${REASON_MODEL}
  --intent-model ${INTENT_MODEL}
  --ranking-score ig
"
REASON_ARGS_FLAT=$(echo $REASON_ARGS)

export SAM2_ROOT="${SAM2_ROOT:-$HOME/miniconda3/envs/smartgrasp/lib/python3.12/site-packages/sam2}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-D3Kd8gupG4HqUgTMsawHZBPmlEolExOmFHgkUkPt6TKuhllT}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://www.highland-api.top/v1}"

PYTHON="$HOME/miniconda3/envs/smartgrasp/bin/python"
[[ -x "$PYTHON" ]] || { echo "❌ Python not found" >&2; exit 1; }

CATEGORY="${1:-}"
shift 2>/dev/null || true

FROM_SCENE=""
SCENE_IDS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_SCENE="$2"; shift 2 ;;
        *) SCENE_IDS="$SCENE_IDS $1"; shift ;;
    esac
done
SCENE_IDS="${SCENE_IDS# }"  # trim leading space

# 日志
mkdir -p logs/ssr
LOG="logs/ssr/run_${CATEGORY:-all}_$(date +%Y%m%d_%H%M%S).log"
exec > "$LOG" 2>&1
echo "Log: $LOG"

# 生成任务清单
"$PYTHON" "$ROOT_DIR/ssr/prepare.py" > /dev/null

# 读取过滤后的任务（用 ||| 分隔，避免 annotation 内空格被切碎）
TASKS=$("$PYTHON" -c "
import json
tasks = json.load(open('ssr/tasks.json'))
cat = '${CATEGORY}'
sids = set('${SCENE_IDS}'.split()) if '${SCENE_IDS}' else set()
from_sid = '${FROM_SCENE}'
for t in tasks:
    if cat and t['category'] != cat: continue
    if from_sid and t['scene_id'] < int(from_sid): continue
    if sids and str(t['scene_id']) not in sids: continue
    print(json.dumps(t))
")

# 逐 scene 循环（Python 处理，避免 bash IFS 切碎 annotation）
"$PYTHON" -c "
import json, subprocess, sys, shutil
from pathlib import Path

ROOT = Path('$ROOT_DIR')
PYTHON = '$PYTHON'
CAT = '${CATEGORY}'
SIDS = set('${SCENE_IDS}'.split()) if '${SCENE_IDS}' else set()
FROM = '${FROM_SCENE}'

tasks = json.load(open('ssr/tasks.json'))
total = 0
ok = 0
fail = 0

for t in tasks:
    if CAT and t['category'] != CAT: continue
    if FROM and t['scene_id'] < int(FROM): continue
    if SIDS and str(t['scene_id']) not in SIDS: continue
    total += 1

for t in tasks:
    if CAT and t['category'] != CAT: continue
    if FROM and t['scene_id'] < int(FROM): continue
    if SIDS and str(t['scene_id']) not in SIDS: continue

    sid = t['scene_id']
    cat = t['category']
    a0  = t['annotations']['0']
    a1  = t['annotations']['1']
    a2  = t['annotations']['2']

    ok += 1
    print(f'\n===== [{ok}/{total}] scene_{sid} ({cat}) =====', flush=True)

    dst = ROOT / 'data' / cat / f'scene_{sid}'
    dst.mkdir(parents=True, exist_ok=True)

    # 1) VLM perception（自动生成 gt/ + perception/）
    print(f'  [1] perception VLM ...', flush=True)
    r = subprocess.run(
        [PYTHON, '-u', '-m', 'perception.run_perception', '--scene-id', str(sid), '--mode', 'vlm']
        + '${PERCEPTION_ARGS_FLAT}'.split(),
        capture_output=True, text=True, cwd=str(ROOT)
    )
    if r.returncode != 0:
        print(f'  [1] perception VLM: FAIL → skip', flush=True)
        print(r.stderr[-300:], flush=True)
        fail += 1
        continue
    print(f'  [1] perception VLM: OK', flush=True)

    # 2) Reason × 3 splits（内部自动跑 intent）
    for split_name, annot in [('0', a0), ('1', a1), ('2', a2)]:
        print(f'  [3] split{split_name}: {annot[:60]}', flush=True)
        r = subprocess.run(
            [PYTHON, '-u', '-m', 'reason.run_reason',
             '--root', 'data', '--scene-id', str(sid),
             '--target-source', 'auto', '--instruction', annot,
             '--scene-root', 'data', '--quiet']
             + '${REASON_ARGS_FLAT}'.split(),
            capture_output=True, text=True, cwd=str(ROOT)
        )
        intent_src = ROOT / 'data' / f'scene_{sid}' / 'intent'
        reason_src = ROOT / 'data' / f'scene_{sid}' / 'reason'
        intent_dst = ROOT / 'data' / f'scene_{sid}' / f'intent_split{split_name}'
        reason_dst = ROOT / 'data' / f'scene_{sid}' / f'reason_split{split_name}'

        if intent_src.exists():
            if intent_dst.exists(): shutil.rmtree(intent_dst)
            shutil.move(str(intent_src), str(intent_dst))
        if reason_src.exists():
            if reason_dst.exists(): shutil.rmtree(reason_dst)
            shutil.move(str(reason_src), str(reason_dst))
        print(f'    split{split_name}: done', flush=True)

    # 3) Organize（gt/ + perception/ + splits → 目标目录）
    print(f'  [3] organize → data/{cat}/scene_{sid}/', flush=True)
    src = ROOT / 'data' / f'scene_{sid}'

    # gt + perception
    for sub in ['gt', 'perception']:
        sub_src = src / sub
        sub_dst = dst / sub
        if sub_src.exists():
            if sub_dst.exists(): shutil.rmtree(sub_dst)
            shutil.move(str(sub_src), str(sub_dst))

    for s in ['0','1','2']:
        for sub in ['intent', 'reason']:
            sd = dst / sub / f'split{s}'
            sd.mkdir(parents=True, exist_ok=True)
            ss = src / f'{sub}_split{s}'
            if ss.exists():
                for f in ss.iterdir():
                    shutil.move(str(f), str(sd / f.name))
                ss.rmdir()

    shutil.rmtree(src, ignore_errors=True)
    print(f'  [3] done', flush=True)

print(f'\n===== DONE: {ok-fail}/{total} OK, {fail} failed =====', flush=True)
"
