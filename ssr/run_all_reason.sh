#!/usr/bin/env bash
# 基于已有 perception，只重跑 intent + reason
# 用法: bash ssr/run_all_reason.sh [category] [scene_ids ...] [--from N] [--query QID]

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

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
LOG="logs/ssr/run_reason_${CATEGORY_STR//,/_}_$(date +%Y%m%d_%H%M%S).log"
exec > "$LOG" 2>&1
echo "Log: $LOG"

# 生成任务清单
"$PYTHON" "$ROOT_DIR/ssr/prepare.py" > /dev/null

# 逐 scene 循环
"$PYTHON" -c "
import json, os, subprocess, sys, shutil
from pathlib import Path

ROOT = Path('$ROOT_DIR')
PYTHON = '$PYTHON'
CATS = set('${CATEGORY_STR}'.split(',')) if '${CATEGORY_STR}' != 'all' else set()
SIDS = set('${SCENE_IDS}'.split()) if '${SCENE_IDS}' else set()
FROM = '${FROM_SCENE}'
QUERY = '${QUERY_ID}'

ALL_CATS = ['easy', 'easy-ambi', 'medium', 'medium-ambi', 'hard', 'hard-ambi']

def find_base_scene_dir(scene_id):
    for cat in ALL_CATS:
        d = ROOT / 'data' / cat / f'scene_{scene_id}'
        if d.is_dir():
            return d
    return None

def is_secondary_query(dir_name):
    return '_q' in dir_name.replace('scene_', '', 1)

# 验证环境变量
_key = os.environ.get('OPENAI_API_KEY', '')
if not _key:
    print('FATAL: OPENAI_API_KEY not set in environment!', flush=True)
    sys.exit(1)
print(f'[env] OPENAI_API_KEY = {_key[:8]}...{_key[-4:]}', flush=True)

def run_one(t, idx, total, is_manual=False):
    sid = t['scene_id']
    cat = t['category']
    dir_name = t.get('directory_name', f'scene_{sid}')
    a0 = t['annotations']['0']
    a1 = t['annotations']['1']
    a2 = t['annotations']['2']
    secondary = is_secondary_query(dir_name)

    label = f'--query {t[\"query_obj_id\"]}' if is_manual else ''
    print(f'\\n===== [{idx}/{total}] {dir_name} ({cat}) {label} =====', flush=True)

    dst = ROOT / 'data' / cat / dir_name
    dst.mkdir(parents=True, exist_ok=True)

    # 确定 perception 来源
    if secondary:
        base_dir = find_base_scene_dir(sid)
        if base_dir is None:
            print(f'  [skip] base scene scene_{sid} not found', flush=True)
            return False
        per_dir = base_dir / 'perception'
    else:
        per_dir = ROOT / 'data' / cat / f'scene_{sid}' / 'perception'

    if not (per_dir / 'summary.json').exists():
        print(f'  [skip] no perception summary in {per_dir} → skip', flush=True)
        return False

    # 如果 secondary，软链接 perception + gt
    if secondary:
        for sub in ['perception', 'gt']:
            base_sub = base_dir / sub
            dst_sub = dst / sub
            if base_sub.exists():
                if dst_sub.is_symlink() or dst_sub.exists():
                    dst_sub.unlink() if dst_sub.is_symlink() else shutil.rmtree(dst_sub)
                rel = os.path.relpath(base_sub, dst)
                os.symlink(rel, str(dst_sub))
                print(f'  [link] {sub}/ → {rel}', flush=True)
        # reason 用 --root data/{cat} 找 data/{cat}/scene_{sid}/perception，
        # 跨类别时该路径不存在，需要临时软链接（文件级，rglob 才能跟随）
        reason_scene = ROOT / 'data' / cat / f'scene_{sid}'
        if not (reason_scene / 'perception').exists():
            reason_scene.mkdir(parents=True, exist_ok=True)
            for sub in ['perception', 'gt']:
                base_sub = base_dir / sub
                tmp_sub = reason_scene / sub
                tmp_sub.mkdir(exist_ok=True)
                for item in base_sub.iterdir():
                    target = tmp_sub / item.name
                    if not target.exists():
                        os.symlink(os.path.relpath(item, tmp_sub), str(target))

    # 清除旧的 intent 和 reason
    for sub in ['intent', 'reason']:
        old = dst / sub
        if old.exists():
            shutil.rmtree(old)
            print(f'  [clean] removed {old}', flush=True)

    # 清理临时 scene 目录
    tmp = ROOT / 'data' / f'scene_{sid}'
    if tmp.exists():
        shutil.rmtree(tmp)

    if not a0.strip() or not a1.strip() or not a2.strip():
        print(f'  [skip] empty annotation → skip', flush=True)
        return False

    # Reason × 3 splits
    for split_name, annot in [('0', a0), ('1', a1), ('2', a2)]:
        print(f'  reason split{split_name}: {annot[:60]}', flush=True)
        r = subprocess.run(
            [PYTHON, '-u', '-m', 'reason.run_reason',
             '--root', f'data/{cat}', '--scene-id', str(sid),
             '--target-source', 'auto', '--instruction', annot,
             '--scene-root', f'data/{cat}', '--quiet']
             + '${REASON_ARGS_FLAT}'.split(),
            capture_output=True, text=True, cwd=str(ROOT),
            env={**os.environ},
        )
        if r.returncode != 0:
            print(f'    split{split_name}: FAIL (rc={r.returncode})', flush=True)
            print(r.stderr[-500:], flush=True)
            return False
        moved_any = False
        for sub in ['intent', 'reason']:
            src = dst / sub
            flat = dst / f'{sub}_split{split_name}'
            if src.exists():
                if flat.exists():
                    shutil.rmtree(flat)
                shutil.move(str(src), str(flat))
                moved_any = True
        if not moved_any:
            print(f'    split{split_name}: no intent/reason produced!', flush=True)
            print(r.stdout[-800:], flush=True)
            print(r.stderr[-800:], flush=True)
            return False
        print(f'    split{split_name}: done', flush=True)

    # 归入 split 子目录
    for s in ['0', '1', '2']:
        for sub in ['intent', 'reason']:
            flat = dst / f'{sub}_split{s}'
            target = dst / sub / f'split{s}'
            if flat.exists():
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(flat), str(target))
    print(f'  done', flush=True)
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
    total = len(pairs)
    for qid, sid in sorted(pairs, key=lambda x: x[1]):
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
    # 剩余 SIDS 走正常流程
    remaining = SIDS - paired_sids
    if remaining:
        for t in tasks:
            if CATS and t['category'] not in CATS: continue
            if FROM and t['scene_id'] < int(FROM): continue
            if str(t['scene_id']) not in remaining: continue
            total += 1
        for t in tasks:
            if CATS and t['category'] not in CATS: continue
            if FROM and t['scene_id'] < int(FROM): continue
            if str(t['scene_id']) not in remaining: continue
            ok += 1
            if not run_one(t, ok, total):
                fail += 1
            fail += 1
        elif not run_one(task, ok, total, is_manual=True):
            fail += 1
else:
    for t in tasks:
        if CATS and t['category'] not in CATS: continue
        if FROM and t['scene_id'] < int(FROM): continue
        if SIDS and str(t['scene_id']) not in SIDS: continue
        total += 1

    for t in tasks:
        if CATS and t['category'] not in CATS: continue
        if FROM and t['scene_id'] < int(FROM): continue
        if SIDS and str(t['scene_id']) not in SIDS: continue
        ok += 1
        if not run_one(t, ok, total):
            fail += 1

print(f'\\n===== DONE: {ok-fail}/{total} OK, {fail} failed =====', flush=True)
"
