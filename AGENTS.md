# SmartGrasp 项目开发约定

## 环境激活

运行任何代码前，必须先激活 conda 环境：

```bash
conda activate smartgrasp
```

## 代理检查

跑程序之前务必检查代理状态。如果不是代理模式，需要先开启代理：

```bash
# 检查当前代理状态
proxy_status

# 如果不是代理模式，开启代理
proxy_on
```
## 分支管理

修改代码时务必在当前分支上面修改，不可更改其他分支

## 运行程序

使用 Shell 脚本 + SLURM 提交任务，不要直接裸跑 Python 脚本。

- Shell 脚本 `run_perception.sh`
- SLURM 作业通过 `sbatch` 提交：

```bash
sbatch run_perception.sh
```

## 代码规范

- 所有变量和函数的命名要可读易理解，尽量不要以补很多后缀的方式打补丁。项目代码风格要一致。
- 代码中要有适当的注释，特别是复杂的逻辑部分，确保其他人能够理解代码的意图。
- 要敢于重构部分代码，保持代码的整洁和可维护性。发现当前代码结构不是最优的时候，要积极重构，不要害怕删除冗余代码或重命名变量以提高代码质量。
- 每次修改 Python 文件后，使用 Pylance 语法检查确保没有引入语法错误。
- 运行前确认smartgrasp环境开启、 `proxy_status` 输出为代理模式

## 测试与验证
- 除非用户主动明确要求测试，不要私自运行项目代码进行测试。
- 除非用户主动明确要求测试，不要私自运行项目代码进行测试。
- 除非用户主动明确要求测试，不要私自运行项目代码进行测试。
- 自己做完代码的修改，也不要自行测试，除非用户明确要求你进行测试。
- 每次测试后主动查看 `.err` 和 `.out` 日志，如果发现报错要主动修改并重新测试，直到没有报错。

### 提交批量任务

```bash
# 单个场景
sbatch run_perception.sh --scene-id 184

# 多个场景（逗号分隔或空格分隔）
sbatch run_perception.sh --scene-ids 184 59 125
```

### 任务监控

- 每个场景处理约 80~100 秒（含 VLM API 调用），批量任务按场景数估算总耗时。
- 使用 `sleep` 间隔检查任务状态，避免频繁轮询：

```bash
# 等待 N 秒后检查进度，任务完成后输出日志
sleep N && tail -10 logs/perception-<jobid>.err && squeue -u $USER | grep <jobid> || echo "JOB DONE"
```

- 优先通过输出目录是否有 `occlusion_graph.json` 判断单场景是否完成：

```bash
ls data/scene_<id>/perception/occlusion_graph.json
```

### 结果验证

```bash
# 查看所有 mask 的面积和来源
python3 -c "
import json
for scene in [184, 59, 125]:
    with open(f'data/scene_{scene}/perception/occlusion_graph.json') as f:
        d = json.load(f)
    for n in d['graph']['nodes']:
        print(f'scene_{scene} mask_{n[\"object_id\"]}: area={n.get(\"mask_area\",0)} backend={n.get(\"segmentation_backend\",\"?\")}')
"
```

- 检查是否有 `sam2_auto_unclaimed_depth_grouped` 类型的小 mask 碎片（正常应全部为 `sam2_auto_review_langsam`）。
- 最小合法 mask 面积阈值为 0.07%（1200×1200 下约 1008px），若有低于此值的 mask 说明过滤失效。

### VLM API 连通性测试

```bash
conda activate smartgrasp && python -u test_vlm_api.py
```
