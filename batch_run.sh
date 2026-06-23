#!/usr/bin/env bash
# ============================================================================
# 批量运行 SmartGrasp Perception，失败继续，生成计时+失败报告
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SCENES=(59 125 222 242 691 823 827)
RESULTS_FILE="$ROOT_DIR/batch_results.txt"
TIMING_FILE="$ROOT_DIR/batch_timing.csv"
FAILURE_FILE="$ROOT_DIR/batch_failures.txt"

echo "scene_id,status,duration_sec,num_nodes,num_edges,error" > "$TIMING_FILE"
echo "" > "$FAILURE_FILE"

total_scenes=${#SCENES[@]}
passed=0
failed=0
declare -a failed_scenes=()

echo "========================================="
echo " Batch Run: ${total_scenes} scenes"
echo "========================================="
echo ""

for scene_id in "${SCENES[@]}"; do
    echo ">>> Scene $scene_id ($((passed + failed + 1))/$total_scenes) ..."
    start_time=$(date +%s)

    rm -rf "data/scene_$scene_id"

    # 运行（输出自动写入 logs/）
    bash run_perception.sh "$scene_id" &>/dev/null
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # 从最新日志提取结果
    latest_log=$(ls -t logs/*.log 2>/dev/null | head -1)
    if [[ $exit_code -eq 0 ]] && grep -q "✅ PASS" "$latest_log" 2>/dev/null; then
        num_nodes=$(grep -oP '"num_nodes": \K\d+' "$latest_log" | head -1)
        num_edges=$(grep -oP '"num_edges": \K\d+' "$latest_log" | head -1)
        echo "    ✅ PASS  (${duration}s, ${num_nodes:-?} nodes, ${num_edges:-?} edges)"
        echo "$scene_id,PASS,$duration,${num_nodes:-0},${num_edges:-0}," >> "$TIMING_FILE"
        passed=$((passed + 1))
    else
        error_line=$(grep -i "Error\|Traceback\|FAILED" "$latest_log" 2>/dev/null | tail -1 | tr ',' ' ' | cut -c1-200)
        echo "    ❌ FAIL  (${duration}s) → $latest_log"
        echo "       $error_line"
        echo "$scene_id,FAIL,$duration,,,$error_line" >> "$TIMING_FILE"
        echo "Scene $scene_id (${duration}s) → $latest_log" >> "$FAILURE_FILE"
        echo "  $error_line" >> "$FAILURE_FILE"
        echo "" >> "$FAILURE_FILE"
        failed_scenes+=("$scene_id")
        failed=$((failed + 1))
    fi
done

# ---- 汇总报告 ----
echo ""
echo "========================================="
echo " Batch Summary"
echo "========================================="
echo "  Total:  $total_scenes"
echo "  Passed: $passed"
echo "  Failed: $failed"
if [[ $failed -gt 0 ]]; then
    echo "  Failed IDs: ${failed_scenes[*]}"
fi
echo ""
echo "Timing: $TIMING_FILE"
echo "Failures: $FAILURE_FILE"

# 打印 timing 表格
echo ""
echo "--- Timing Report ---"
column -t -s',' "$TIMING_FILE"
