# SmartGrasp - Execution Module (执行模块)

本目录包含了 SmartGrasp 视觉语言任务导向抓取系统（Vision-Language Task-Oriented Grasping）的核心**执行模块（Execution Module）**。

执行模块作为机器人的“算法大脑”，负责接收上游感知（Perception）与推理（Reasoning）模块传来的信号与目标/遮挡物 ID。本模块会根据场景的遮挡状态，精准调用底层 3D 视觉算法与 GraspNet 大模型，最终输出物理机械臂可直接执行的 6-DoF 控制指令。

---

## 📂 目录结构与文件说明 (Directory Structure)

`execution/`
* `execution_api.py` : [核心API] 完全可见分支：目标物体的 6D 抓取
* `reveal_api.py` : [核心API] 遮挡排雷分支：障碍物的微动推开与闭环触发
* `grasp_generator.py` : [算法层] GraspNet 大模型调用与最优位姿筛选计算
* `pointcloud_utils.py` : [工具层] 3D 点云处理（RGB-D + Mask 转换局部点云）
* `test_pipeline.py` : [测试层] 单目标/Mock 数据管线联调测试脚本
* `batch_test.py` : [测试层] 批量自动化验证脚本（遍历测试真实场景数据）
* `results/` : [输出层] 存放抓取测试生成的位姿结果与日志
* `__pycache__/` : Python 编译缓存目录

---

## 🧠 核心功能模块 (Core Modules)

本模块严格按照《SmartGrasp 项目架构》分为两大执行分支，以应对不同的场景复杂度：

### 1. 抓取模块 (Fully Visible 分支) -> `execution_api.py`
* **适用场景**：当目标物体完全可见（处于有向遮挡关系图的最顶层）时调用。
* **工作流**：
  1. 接收上游传来的 Target ID 和 2D Mask。
  2. 调用 `pointcloud_utils.py`，结合相机内参（Intrinsics）将深度图与 Mask 反投影为干净的**局部 3D 点云**。
  3. 调用 `grasp_generator.py`（封装了预训练的 GraspNet-1Billion 网络），生成 1024 个候选姿态。
  4. 根据物理稳定性（Force-closure 得分等）筛选出 **Top-1 最优 6-DoF 抓取位姿**（包含 Rotation, Translation, Width）。

### 2. 微动探测模块 (Occluded 分支) -> `reveal_api.py`
* **适用场景**：当目标物体被部分遮挡（Partially Occluded）或完全不可见（Fully Occluded）时调用。
* **工作流**：
  1. 接收推理模块基于信息增益（InfoGain）算出的“得分最高的遮挡物 ID”（挡路石）及其 3D 几何中心点。
  2. 采用极简的数学控制策略替代复杂大模型：输出默认垂直向下的夹爪姿态，并针对遮挡物在 X 轴执行 3-5 厘米的**轻微拨动（push）**或**小幅抓放（pick_and_place）**，以维持复杂堆叠场景的物理稳定性。
  3. 动作生成后，向全局输出 `request_reloop: True` 信号，触发系统重新拍摄 RGB-D 照片进入下一轮闭环（RE-LOOPS）。

---

## 🛠️ 测试与验证 (Testing & Validation)

在不依赖外部 PyBullet 物理仿真环境的情况下，您可以直接在本目录下对算法的大脑逻辑进行独立闭环测试。

**环境要求**：请确保处于 `smartgrasp` Conda 虚拟环境中。

### 1. 运行单步管线测试
测试基于伪造数据（Mock RGB-D & Mask）的 GraspNet 调用流程：
`python test_pipeline.py`
*(成功后会在终端输出被筛选出的最优 6D 抓取位姿参数)*

### 2. 运行批量验证测试
测试多张真实物体 Mask 在算法管线中的鲁棒性：
`python batch_test.py`
*(脚本会自动遍历处理多组掩码数据，结果保存在 `results/` 目录下)*

### 3. 测试微动探测逻辑
测试排雷推开坐标计算与闭环信号生成：
`python reveal_api.py`

---

## 🔗 后续集成 (Integration Note)

本目录为**纯算法逻辑层**。在项目的第 3 个月（Month 3）进行全系统闭环测试时，本目录下的 API 将被主控流水线（如 `graspnet-workspace/scripts/demo_closed_loop.py`）动态导入。主程序将根据感知层传来的 `is_occluded` 状态，对 `execution_api.py` 和 `reveal_api.py` 进行交通路由，并在 PyBullet 仿真引擎中渲染最终的机械臂 3D 物理执行动画。