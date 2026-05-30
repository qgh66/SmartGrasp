"""
Simulation 模块

基于 PyBullet 的 GraspNet 抓取仿真验证框架。

架构:
  scene.py       - PyBullet 场景管理（加载物体、桌面、光照）
  camera.py      - 虚拟 RGB-D 相机（拍摄、生成点云）
  gripper.py     - 平行二指夹爪模型（开合控制）
  evaluator.py   - 抓取执行与物理评估
  run_sim.py     - 主入口：串联感知→推理→执行→评估

数据流:
  PyBullet场景 → 虚拟相机拍摄 → 点云 → GraspNet推理 → GraspGroup → 逐抓取执行 → 成功/失败标签
"""
