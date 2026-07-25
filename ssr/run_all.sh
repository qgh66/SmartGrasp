#!/usr/bin/env bash
# 逐 scene 完整处理：perception → gt → intent ×3 → reason ×3 → 放入分类目录
# 用法:
#   bash ssr/run_all.sh                  # 全部 6 类
#   bash ssr/run_all.sh --all            # 全部 6 类
#   bash ssr/run_all.sh easy             # 只跑 easy
#   bash ssr/run_all.sh easy 0 20        # 只跑指定 scene
#   bash ssr/run_all.sh --from 1556      # 全部类别，从 scene_1556 开始
#   bash ssr/run_all.sh hard-ambi --from 1556  # hard-ambi，从 scene_1556 开始
#   bash ssr/run_all.sh hard-ambi --query 4 1449  # scene_1449 的 query 4 → hard-ambi（_q4 后缀，复用 perception）

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
REASON_MODEL="gpt-5.5"
INTENT_MODEL="gpt-5.5"
REASON_ARGS="
  --model ${REASON_MODEL}
  --intent-model ${INTENT_MODEL}
  --ranking-score ig
"
REASON_ARGS_FLAT=$(echo $REASON_ARGS)

export SAM2_ROOT="${SAM2_ROOT:-$HOME/miniconda3/envs/smartgrasp/lib/python3.12/site-packages/sam2}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$(python3 -c "import json;print(json.load(open('$ROOT_DIR/api_config.json'))['api_key'])")}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$(python3 -c "import json;print(json.load(open('$ROOT_DIR/api_config.json'))['base_url'])")}"

PYTHON="$HOME/miniconda3/envs/smartgrasp/bin/python"
[[ -x "$PYTHON" ]] || { echo "❌ Python not found" >&2; exit 1; }

CATEGORIES=()
FROM_SCENE=""
SCENE_IDS=""
QUERY_ID=""
QUERY_PAIRS=""  # "QID,SID QID,SID ..." each --query binds to exactly 1 scene
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) CATEGORIES=(); shift ;;
        --from) FROM_SCENE="$2"; shift 2 ;;
        --query)
            q="$2"
            shift 2
            # 只绑定紧跟的一个 scene
            if [[ $# -gt 0 && "$1" != -* ]]; then
                QUERY_PAIRS="$QUERY_PAIRS $q,$1"
                shift
            fi
            ;;
        easy|easy-ambi|medium|medium-ambi|hard|hard-ambi)
            CATEGORIES+=("$1"); shift ;;
        *)
            SCENE_IDS="$SCENE_IDS $1"
            shift
            ;;
    esac
done
SCENE_IDS="${SCENE_IDS# }"
if [[ -n "$QUERY_PAIRS" ]]; then
    QUERY_ID="${QUERY_PAIRS# }"
fi
CATEGORY_STR=$(IFS=,; echo "${CATEGORIES[*]}")
CATEGORY_STR="${CATEGORY_STR:-all}"

# 日志
mkdir -p logs/ssr
LOG="logs/ssr/run_${CATEGORY_STR//,/_}_$(date +%Y%m%d_%H%M%S).log"
exec > "$LOG" 2>&1
echo "Log: $LOG"

# 生成任务清单
"$PYTHON" "$ROOT_DIR/ssr/prepare.py" > /dev/null

# 读取过滤后的任务（用 ||| 分隔，避免 annotation 内空格被切碎）
TASKS=$("$PYTHON" -c "
import json
tasks = json.load(open('ssr/tasks.json'))
cat_str = '${CATEGORY_STR}'
cats = set(cat_str.split(',')) if cat_str and cat_str != 'all' else set()
sids = set('${SCENE_IDS}'.split()) if '${SCENE_IDS}' else set()
from_sid = '${FROM_SCENE}'
for t in tasks:
    if cats and t['category'] not in cats: continue
    if from_sid and t['scene_id'] < int(from_sid): continue
    if sids and str(t['scene_id']) not in sids: continue
    print(json.dumps(t))
")

# 逐 scene 循环（Python 处理，避免 bash IFS 切碎 annotation）
"$PYTHON" -c "
import json, subprocess, sys, shutil, os
from pathlib import Path

ROOT = Path('$ROOT_DIR')
PYTHON = '$PYTHON'
CATS = set('${CATEGORY_STR}'.split(',')) if '${CATEGORY_STR}' != 'all' else set()
SIDS = set('${SCENE_IDS}'.split()) if '${SCENE_IDS}' else set()
FROM = '${FROM_SCENE}'
QUERY = '${QUERY_ID}'

def run_one(t, idx, total, is_manual=False):
    sid = t['scene_id']
    cat = t['category']
    dir_name = t.get('directory_name', f'scene_{sid}')
    a0 = t['annotations']['0']
    a1 = t['annotations']['1']
    a2 = t['annotations']['2']

    label = f'--query {t[\"query_obj_id\"]}' if is_manual else ''
    print(f'\\n===== [{idx}/{total}] {dir_name} ({cat}) {label} =====', flush=True)

    # 清除临时 scene_X 目录
    tmp_src = ROOT / 'data' / f'scene_{sid}'
    if tmp_src.exists():
        shutil.rmtree(tmp_src)

    # 清除旧目标目录（重新跑总是清干净）
    dst = ROOT / 'data' / cat / dir_name
    if dst.exists():
        shutil.rmtree(dst)
        print(f'  [clean] removed {dst}', flush=True)
    dst.mkdir(parents=True, exist_ok=True)

    # 完整 perception
    print(f'  [1] perception VLM ...', flush=True)
    r = subprocess.run(
        [PYTHON, '-u', '-m', 'perception.run_perception', '--scene-id', str(sid), '--mode', 'vlm']
        + '${PERCEPTION_ARGS_FLAT}'.split(),
        capture_output=True, text=True, cwd=str(ROOT)
    )
    if r.returncode != 0:
        print(f'  [1] perception VLM: FAIL → skip', flush=True)
        print(r.stderr[-300:], flush=True)
        return False
    print(f'  [1] perception VLM: OK', flush=True)

    # Reason × 3 splits
    for split_name, annot in [('0', a0), ('1', a1), ('2', a2)]:
        print(f'  [reason] split{split_name}: {annot[:60]}', flush=True)
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

    # Organize
    print(f'  [organize] → data/{cat}/{dir_name}/', flush=True)
    src = ROOT / 'data' / f'scene_{sid}'

    for sub in ['gt', 'perception']:
        sub_src = src / sub
        sub_dst = dst / sub
        if sub_src.exists():
            if sub_dst.exists() and not sub_dst.is_symlink():
                shutil.rmtree(sub_dst)
            if not sub_dst.exists():
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
    print(f'  [organize] done', flush=True)
    return True

# ── 主流程 ──
tasks = json.load(open('ssr/tasks.json'))
total = 0
ok = 0
fail = 0

if QUERY:
    # 解析 --query QID SID 对
    pairs = []
    paired_sids = set()
    for token in QUERY.split():
        qid_str, sid_str = token.split(',')
        pairs.append((int(qid_str), int(sid_str)))
        paired_sids.add(sid_str)
    # 先处理 standalone SIDS（只匹配 primary，排除 _qN）
    if SIDS:
        for t in tasks:
            if '_q' in t.get('directory_name', ''): continue
            if CATS and t['category'] not in CATS: continue
            if FROM and t['scene_id'] < int(FROM): continue
            if str(t['scene_id']) not in SIDS: continue
            total += 1
    total += len(pairs)
    if SIDS:
        for t in tasks:
            if '_q' in t.get('directory_name', ''): continue
            if CATS and t['category'] not in CATS: continue
            if FROM and t['scene_id'] < int(FROM): continue
            if str(t['scene_id']) not in SIDS: continue
            ok += 1
            if not run_one(t, ok, total):
                fail += 1
    # 再跑 query 对（保持命令行顺序）
    for qid, sid in pairs:
        ok += 1
        task = None
        for t in tasks:
            if t['scene_id'] == sid and t['query_obj_id'] == qid:
                task = t
                break
        if task is None:
            print(f'  [skip] scene_{sid} qid={qid}: not in tasks.json', flush=True)
            fail += 1
        elif not run_one(task, ok, total, is_manual=True):
            fail += 1
else:
    for t in tasks:
        if SIDS and '_q' in t.get('directory_name', ''): continue
        if CATS and t['category'] not in CATS: continue
        if FROM and t['scene_id'] < int(FROM): continue
        if SIDS and str(t['scene_id']) not in SIDS: continue
        total += 1

    for t in tasks:
        if SIDS and '_q' in t.get('directory_name', ''): continue
        if CATS and t['category'] not in CATS: continue
        if FROM and t['scene_id'] < int(FROM): continue
        if SIDS and str(t['scene_id']) not in SIDS: continue
        ok += 1
        if not run_one(t, ok, total):
            fail += 1

print(f'\\n===== DONE: {ok-fail}/{total} OK, {fail} failed =====', flush=True)
"
