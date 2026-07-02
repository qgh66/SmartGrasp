# SmartGrasp 项目开发约定

## 环境激活

运行任何代码前，必须先激活 conda 环境：

```bash
conda activate smartgrasp
```


## 分支管理

后续所有代码修改都在 `feat/grasp_realworld` 分支中进行。修改前先确认当前分支：

```bash
git branch --show-current
```

如果不在 `feat/grasp_realworld`，先切回该分支；不可在其他分支上修改代码。

## 运行程序

当前服务器不强制遵循 SLURM 提交流程。需要运行程序时，先激活 `smartgrasp`
环境并确认代理状态，然后按当前任务选择本地 Shell 脚本或直接入口命令运行。
除非用户明确要求通过 SLURM 提交，否则不要默认使用 `sbatch`。

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

### 长任务监控

- 每个场景处理约 80~100 秒（含 VLM API 调用），批量任务按场景数估算总耗时。
- 如果用户明确要求运行长任务，使用合理的 `sleep` 间隔检查进度，避免频繁轮询。
- 如果通过脚本产生 `.err` 和 `.out` 日志，测试后主动查看日志；如果发现报错，主动修改并按用户要求重新测试。

- 优先通过输出目录是否有 `occlusion_graph.json` 判断单场景是否完成；当前感知输出目录格式为
  `data/integrated_runs/scene_<scene_id>_query_<query_obj_id>_<point_source>/`：

```bash
ls data/integrated_runs/scene_<scene_id>_query_<query_obj_id>_<point_source>/occlusion_graph.json
```

### 结果验证

```bash
# 查看所有 mask 的面积和来源
python3 -c "
import json
from pathlib import Path
for path in sorted(Path('data/integrated_runs').glob('scene_*/occlusion_graph.json')):
    with path.open() as f:
        d = json.load(f)
    for n in d['graph']['nodes']:
        print(f'{path.parent.name} node_{n[\"node_id\"]}: area={n.get(\"mask_area\", 0)} label={n.get(\"label\", \"?\")}')
"
```

- 检查是否有 `sam2_auto_unclaimed_depth_grouped` 类型的小 mask 碎片（正常应全部为 `sam2_auto_review_langsam`）。
- 最小合法 mask 面积阈值为 0.07%（1200×1200 下约 1008px），若有低于此值的 mask 说明过滤失效。

### VLM API 连通性测试

```bash
conda activate smartgrasp && python -u test_vlm_api.py
```
