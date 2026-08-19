# SmartGrasp FreeGrasp-style 六类实验结果

- Run ID: `job_manual`
- 生成时间: `2026-08-19T21:39:56+08:00`
- 结果清单: `/home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/run_manifest.tsv`
- 目标 mask 最小 IoU: `0.5`
- 执行模式: `perception_reason_validation_and_delete`
- GraspNet: 禁用
- 机械臂/Push: 禁用
- PyBullet GUI/录像: 禁用

## 六类结果汇总

| 条件 | 场景 | 完成 episode | episode 全通过 | SSR | RSR | 核验后删除 |
|---|---|---:|---:|---:|---:|---:|
| 低复杂度 / w/o ambiguity | 平铺无遮挡，目标类别在场景中唯一 | 9/9 | 8/9 (88.9%) | 8/9 (88.9%) | 8/9 (88.9%) | 8/9 (88.9%) |
| 低复杂度 / with ambiguity | 平铺无遮挡，同时存在小/中/大三个夹具 | 9/9 | 4/9 (44.4%) | 9/9 (100.0%) | 4/9 (44.4%) | 4/9 (44.4%) |
| 中复杂度 / w/o ambiguity | 12 物体堆叠场景，目标是可见上层唯一类别 | 9/9 | 6/9 (66.7%) | 9/9 (100.0%) | 6/9 (66.7%) | 6/9 (66.7%) |
| 中复杂度 / with ambiguity | 12 物体堆叠场景，上下层存在同类实例 | 9/9 | 5/9 (55.6%) | 5/9 (55.6%) | 5/9 (55.6%) | 5/9 (55.6%) |
| 高复杂度 / w/o ambiguity | 先删除上层双色锤，再重新观察唯一的十字螺丝刀 | 3/3 | 3/3 (100.0%) | 6/6 (100.0%) | 6/6 (100.0%) | 6/6 (100.0%) |
| 高复杂度 / with ambiguity | 先删除电钻，再在另一块电池的干扰下定位被揭露电池 | 3/3 | 0/3 (0.0%) | 3/6 (50.0%) | 0/6 (0.0%) | 0/6 (0.0%) |

SSR 按 `objects[*].perception_correct` 统计，RSR 按 `objects[*].reason_correct` 统计。本次二者都使用上方记录的 mask IoU 阈值。“核验后删除”要求 Perception 和 Reason 同时正确。

## 逐 episode 结果

| 条件 | Episode | 提示词编号 | 提示词模板 | 实际展开指令 | Perception | Reason | 删除 | 状态 | JSON |
|---|---|---:|---|---|---:|---:|---:|---|---|
| 低复杂度 / w/o ambiguity | adjustable_wrench | 1 | 抓取{target} | adjustable_wrench: 抓取活动扳手 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_without_ambiguity__adjustable_wrench__r1.json |
| 低复杂度 / w/o ambiguity | adjustable_wrench | 2 | 请找到{target}并将它取出 | adjustable_wrench: 请找到活动扳手并将它取出 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_without_ambiguity__adjustable_wrench__r2.json |
| 低复杂度 / w/o ambiguity | adjustable_wrench | 3 | 目标物体是{target}，请将它抓取出来 | adjustable_wrench: 目标物体是活动扳手，请将它抓取出来 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_without_ambiguity__adjustable_wrench__r3.json |
| 低复杂度 / w/o ambiguity | power_drill | 1 | 抓取{target} | power_drill: 抓取电钻 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_without_ambiguity__power_drill__r1.json |
| 低复杂度 / w/o ambiguity | power_drill | 2 | 请找到{target}并将它取出 | power_drill: 请找到电钻并将它取出 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_without_ambiguity__power_drill__r2.json |
| 低复杂度 / w/o ambiguity | power_drill | 3 | 目标物体是{target}，请将它抓取出来 | power_drill: 目标物体是电钻，请将它抓取出来 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_without_ambiguity__power_drill__r3.json |
| 低复杂度 / w/o ambiguity | battery | 1 | 抓取{target} | battery: 抓取电池 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_without_ambiguity__battery__r1.json |
| 低复杂度 / w/o ambiguity | battery | 2 | 请找到{target}并将它取出 | battery: 请找到电池并将它取出 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_without_ambiguity__battery__r2.json |
| 低复杂度 / w/o ambiguity | battery | 3 | 目标物体是{target}，请将它抓取出来 | battery: 目标物体是电池，请将它抓取出来 | 0/1 | 0/1 | 0/1 | 未全部通过: battery: max_task_rounds_reached | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_without_ambiguity__battery__r3.json |
| 低复杂度 / with ambiguity | small_clamp | 1 | 抓取{target} | small_clamp: 抓取小号夹具 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_with_ambiguity__small_clamp__r1.json |
| 低复杂度 / with ambiguity | small_clamp | 2 | 请找到{target}并将它取出 | small_clamp: 请找到小号夹具并将它取出 | 1/1 | 0/1 | 0/1 | 未全部通过: small_clamp: max_task_rounds_reached | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_with_ambiguity__small_clamp__r2.json |
| 低复杂度 / with ambiguity | small_clamp | 3 | 目标物体是{target}，请将它抓取出来 | small_clamp: 目标物体是小号夹具，请将它抓取出来 | 1/1 | 0/1 | 0/1 | 未全部通过: small_clamp: max_task_rounds_reached | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_with_ambiguity__small_clamp__r3.json |
| 低复杂度 / with ambiguity | medium_clamp | 1 | 抓取{target} | medium_clamp: 抓取中号夹具 | 1/1 | 0/1 | 0/1 | 未全部通过: medium_clamp: max_task_rounds_reached | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_with_ambiguity__medium_clamp__r1.json |
| 低复杂度 / with ambiguity | medium_clamp | 2 | 请找到{target}并将它取出 | medium_clamp: 请找到中号夹具并将它取出 | 1/1 | 0/1 | 0/1 | 未全部通过: medium_clamp: max_task_rounds_reached | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_with_ambiguity__medium_clamp__r2.json |
| 低复杂度 / with ambiguity | medium_clamp | 3 | 目标物体是{target}，请将它抓取出来 | medium_clamp: 目标物体是中号夹具，请将它抓取出来 | 1/1 | 0/1 | 0/1 | 未全部通过: medium_clamp: max_task_rounds_reached | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_with_ambiguity__medium_clamp__r3.json |
| 低复杂度 / with ambiguity | large_clamp | 1 | 抓取{target} | large_clamp: 抓取大号夹具 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_with_ambiguity__large_clamp__r1.json |
| 低复杂度 / with ambiguity | large_clamp | 2 | 请找到{target}并将它取出 | large_clamp: 请找到大号夹具并将它取出 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_with_ambiguity__large_clamp__r2.json |
| 低复杂度 / with ambiguity | large_clamp | 3 | 目标物体是{target}，请将它抓取出来 | large_clamp: 目标物体是大号夹具，请将它抓取出来 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/low_with_ambiguity__large_clamp__r3.json |
| 中复杂度 / w/o ambiguity | power_drill_cover_a | 1 | 抓取{target} | power_drill_cover_a: 抓取电钻 | 1/1 | 0/1 | 0/1 | 未全部通过: power_drill_cover_a: max_task_rounds_reached | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_without_ambiguity__power_drill_cover_a__r1.json |
| 中复杂度 / w/o ambiguity | power_drill_cover_a | 2 | 请找到{target}并将它取出 | power_drill_cover_a: 请找到电钻并将它取出 | 1/1 | 0/1 | 0/1 | 未全部通过: power_drill_cover_a: max_task_rounds_reached | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_without_ambiguity__power_drill_cover_a__r2.json |
| 中复杂度 / w/o ambiguity | power_drill_cover_a | 3 | 目标物体是{target}，请将它抓取出来 | power_drill_cover_a: 目标物体是电钻，请将它抓取出来 | 1/1 | 0/1 | 0/1 | 未全部通过: power_drill_cover_a: max_task_rounds_reached | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_without_ambiguity__power_drill_cover_a__r3.json |
| 中复杂度 / w/o ambiguity | adjustable_wrench_cover_b | 1 | 抓取{target} | adjustable_wrench_cover_b: 抓取上层的活动扳手 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_without_ambiguity__adjustable_wrench_cover_b__r1.json |
| 中复杂度 / w/o ambiguity | adjustable_wrench_cover_b | 2 | 请找到{target}并将它取出 | adjustable_wrench_cover_b: 请找到上层的活动扳手并将它取出 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_without_ambiguity__adjustable_wrench_cover_b__r2.json |
| 中复杂度 / w/o ambiguity | adjustable_wrench_cover_b | 3 | 目标物体是{target}，请将它抓取出来 | adjustable_wrench_cover_b: 目标物体是上层的活动扳手，请将它抓取出来 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_without_ambiguity__adjustable_wrench_cover_b__r3.json |
| 中复杂度 / w/o ambiguity | two_color_hammer_cover_d | 1 | 抓取{target} | two_color_hammer_cover_d: 抓取上层的双色锤 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_without_ambiguity__two_color_hammer_cover_d__r1.json |
| 中复杂度 / w/o ambiguity | two_color_hammer_cover_d | 2 | 请找到{target}并将它取出 | two_color_hammer_cover_d: 请找到上层的双色锤并将它取出 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_without_ambiguity__two_color_hammer_cover_d__r2.json |
| 中复杂度 / w/o ambiguity | two_color_hammer_cover_d | 3 | 目标物体是{target}，请将它抓取出来 | two_color_hammer_cover_d: 目标物体是上层的双色锤，请将它抓取出来 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_without_ambiguity__two_color_hammer_cover_d__r3.json |
| 中复杂度 / with ambiguity | flat_screwdriver_cover_c | 1 | 抓取{target} | flat_screwdriver_cover_c: 抓取上层的一字螺丝刀 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_with_ambiguity__flat_screwdriver_cover_c__r1.json |
| 中复杂度 / with ambiguity | flat_screwdriver_cover_c | 2 | 请找到{target}并将它取出 | flat_screwdriver_cover_c: 请找到上层的一字螺丝刀并将它取出 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_with_ambiguity__flat_screwdriver_cover_c__r2.json |
| 中复杂度 / with ambiguity | flat_screwdriver_cover_c | 3 | 目标物体是{target}，请将它抓取出来 | flat_screwdriver_cover_c: 目标物体是上层的一字螺丝刀，请将它抓取出来 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_with_ambiguity__flat_screwdriver_cover_c__r3.json |
| 中复杂度 / with ambiguity | battery_cover_e | 1 | 抓取{target} | battery_cover_e: 抓取上层的电池 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_with_ambiguity__battery_cover_e__r1.json |
| 中复杂度 / with ambiguity | battery_cover_e | 2 | 请找到{target}并将它取出 | battery_cover_e: 请找到上层的电池并将它取出 | 0/1 | 0/1 | 0/1 | 未全部通过: battery_cover_e: max_task_rounds_reached | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_with_ambiguity__battery_cover_e__r2.json |
| 中复杂度 / with ambiguity | battery_cover_e | 3 | 目标物体是{target}，请将它抓取出来 | battery_cover_e: 目标物体是上层的电池，请将它抓取出来 | 1/1 | 1/1 | 1/1 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_with_ambiguity__battery_cover_e__r3.json |
| 中复杂度 / with ambiguity | medium_clamp_cover_f | 1 | 抓取{target} | medium_clamp_cover_f: 抓取上层的中号夹具 | 0/1 | 0/1 | 0/1 | 未全部通过: medium_clamp_cover_f: max_task_rounds_reached | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_with_ambiguity__medium_clamp_cover_f__r1.json |
| 中复杂度 / with ambiguity | medium_clamp_cover_f | 2 | 请找到{target}并将它取出 | medium_clamp_cover_f: 请找到上层的中号夹具并将它取出 | 0/1 | 0/1 | 0/1 | 未全部通过: medium_clamp_cover_f: max_task_rounds_reached | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_with_ambiguity__medium_clamp_cover_f__r2.json |
| 中复杂度 / with ambiguity | medium_clamp_cover_f | 3 | 目标物体是{target}，请将它抓取出来 | medium_clamp_cover_f: 目标物体是上层的中号夹具，请将它抓取出来 | 0/1 | 0/1 | 0/1 | 未全部通过: medium_clamp_cover_f: max_task_rounds_reached | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/medium_with_ambiguity__medium_clamp_cover_f__r3.json |
| 高复杂度 / w/o ambiguity | hammer_then_phillips | 1 | 抓取{target} | two_color_hammer_cover_d: 抓取上层的双色锤<br>phillips_screwdriver_base_d: 抓取画面左上方的十字螺丝刀 | 2/2 | 2/2 | 2/2 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/high_without_ambiguity__hammer_then_phillips__r1.json |
| 高复杂度 / w/o ambiguity | hammer_then_phillips | 2 | 请找到{target}并将它取出 | two_color_hammer_cover_d: 请找到上层的双色锤并将它取出<br>phillips_screwdriver_base_d: 请找到画面左上方的十字螺丝刀并将它取出 | 2/2 | 2/2 | 2/2 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/high_without_ambiguity__hammer_then_phillips__r2.json |
| 高复杂度 / w/o ambiguity | hammer_then_phillips | 3 | 目标物体是{target}，请将它抓取出来 | two_color_hammer_cover_d: 目标物体是上层的双色锤，请将它抓取出来<br>phillips_screwdriver_base_d: 目标物体是画面左上方的十字螺丝刀，请将它抓取出来 | 2/2 | 2/2 | 2/2 | 通过 | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/high_without_ambiguity__hammer_then_phillips__r3.json |
| 高复杂度 / with ambiguity | drill_then_battery | 1 | 抓取{target} | power_drill_cover_a: 抓取电钻<br>battery_fully_occluded_a: 抓取电钻下面的电池 | 1/2 | 0/2 | 0/2 | 未全部通过: power_drill_cover_a: max_task_rounds_reached; battery_fully_occluded_a: max_task_rounds_reached | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/high_with_ambiguity__drill_then_battery__r1.json |
| 高复杂度 / with ambiguity | drill_then_battery | 2 | 请找到{target}并将它取出 | power_drill_cover_a: 请找到电钻并将它取出<br>battery_fully_occluded_a: 请找到电钻下面的电池并将它取出 | 1/2 | 0/2 | 0/2 | 未全部通过: power_drill_cover_a: max_task_rounds_reached; battery_fully_occluded_a: max_task_rounds_reached | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/high_with_ambiguity__drill_then_battery__r2.json |
| 高复杂度 / with ambiguity | drill_then_battery | 3 | 目标物体是{target}，请将它抓取出来 | power_drill_cover_a: 目标物体是电钻，请将它抓取出来<br>battery_fully_occluded_a: 目标物体是电钻下面的电池，请将它抓取出来 | 1/2 | 0/2 | 0/2 | 未全部通过: power_drill_cover_a: max_task_rounds_reached; battery_fully_occluded_a: max_task_rounds_reached | /home/admin128/qiuguanhe/Simulation/SmartGrasp/graspnet-workspace/results/freegrasp_style/job_manual/high_with_ambiguity__drill_then_battery__r3.json |

## 作业失败记录

无。
