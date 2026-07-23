#!/usr/bin/env bash
# 基于已有 perception，只重跑 intent + reason
# 用法: bash ssr/run_all_reason.sh [category] [scene_ids ...] [--from N]

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
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-1xiLt7t2zMJOv8YyOoS9zZqSk2FCoVI3Bd4j7wduM4ajjRvK}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://yunwu.ai/v1}"

PYTHON="$HOME/miniconda3/envs/smartgrasp/bin/python"
[[ -x "$PYTHON" ]] || { echo "❌ Python not found" >&2; exit 1; }

CATEGORY=""
FROM_SCENE=""
SCENE_IDS=""
CATEGORY_SET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) CATEGORY=""; CATEGORY_SET="1"; shift ;;
        --from) FROM_SCENE="$2"; shift 2 ;;
        *) 
            if [[ -z "$CATEGORY_SET" && "$1" != --* ]]; then
                CATEGORY="$1"; CATEGORY_SET="1"
            else
                SCENE_IDS="$SCENE_IDS $1"
            fi
            shift
            ;;
    esac
done
SCENE_IDS="${SCENE_IDS# }"

# 日志
mkdir -p logs/ssr
LOG="logs/ssr/run_reason_${CATEGORY:-all}_$(date +%Y%m%d_%H%M%S).log"
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
CAT = '${CATEGORY}'
SIDS = set('${SCENE_IDS}'.split()) if '${SCENE_IDS}' else set()
FROM = '${FROM_SCENE}'

tasks = json.load(open('ssr/tasks.json'))
total = 0
ok = 0
fail = 0

# 验证环境变量
_key = os.environ.get('OPENAI_API_KEY', '')
if not _key:
    print('FATAL: OPENAI_API_KEY not set in environment!', flush=True)
    sys.exit(1)
print(f'[env] OPENAI_API_KEY = {_key[:8]}...{_key[-4:]}', flush=True)

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

    # 检查 perception 是否存在
    per_dir = ROOT / 'data' / cat / f'scene_{sid}' / 'perception'
    if not (per_dir / 'summary.json').exists():
        print(f'  [skip] no perception summary → skip', flush=True)
        fail += 1
        continue

    # 清除旧的 intent 和 reason
    dst = ROOT / 'data' / cat / f'scene_{sid}'
    for sub in ['intent', 'reason']:
        old = dst / sub
        if old.exists():
            shutil.rmtree(old)
            print(f'  [clean] removed {old}', flush=True)

    # 也清理临时 scene 目录
    tmp = ROOT / 'data' / f'scene_{sid}'
    if tmp.exists():
        shutil.rmtree(tmp)

    # 验证 annotation 不为空
    if not a0.strip() or not a1.strip() or not a2.strip():
        print(f'  [skip] empty annotation in task (a0={repr(a0[:30])}) → skip', flush=True)
        fail += 1
        continue

    # Reason × 3 splits（--root data/{cat} 直接从分类目录读 perception）
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
            fail += 1
            continue
        # reason 输出在 dst/intent/ 和 dst/reason/，移到临时 flat 目录
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
            fail += 1
            continue
        print(f'    split{split_name}: done', flush=True)

    # 把 flat 目录归入 intent/splitN/ 和 reason/splitN/
    for s in ['0', '1', '2']:
        for sub in ['intent', 'reason']:
            flat = dst / f'{sub}_split{s}'
            target = dst / sub / f'split{s}'
            if flat.exists():
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(flat), str(target))
    print(f'  done', flush=True)

print(f'\n===== DONE: {ok-fail}/{total} OK, {fail} failed =====', flush=True)
"
