# Reveal Push 仿真说明与验证方法

## 1. 当前默认 Push 仿真在做什么

当前默认模式使用 PyBullet 创建桌面、物体和夹爪，并使用虚拟相机拍摄该
虚拟场景。

完整流程是：

```text
加载 PyBullet 小方块或指定 OBJ
        |
        v
虚拟相机拍摄 PyBullet 场景
        |
        v
生成虚拟 RGB、Depth 和点云
        |
        v
reveal_api.py 生成沿世界坐标系 +X 的 Push 计划
        |
        v
PyBullet 夹爪接近并推动物体
        |
        v
记录物体和夹爪的逐帧位姿
        |
        v
Dash 网页读取结果并回放
```

因此，当前默认模式中的场景、相机、RGB-D、点云和推动过程都是虚拟的。

## 2. 点云在当前 Push 中的作用

虚拟相机会根据 PyBullet 场景生成 RGB、深度图和点云。但是当前点云主要
用于结果保存和网页显示，并没有参与 Push 接触位置或推动方向的计算。

当前 Push 接触位置由 PyBullet 中物体的 AABB 包围盒计算：

```text
物体 AABB
  -> 找到 X 方向待推动表面
  -> 计算夹爪接触位姿
  -> 夹爪从表面外侧接近
  -> 发生 PyBullet 碰撞
  -> 沿 +X 推动物体
```

`reveal_api.py` 当前只负责输出：

- 遮挡物中心。
- 固定的世界坐标系 `+X` 推动方向。
- 默认 `0.05 m` 推动距离。
- 竖直下探的夹爪姿态。
- 推动后的目标位置。
- `request_reloop=True` 闭环请求。

它目前不会根据点云、遮挡关系或桌面边界自动选择最佳 Push 方向。

当前 Push 姿态为：

```text
夹爪局部 X 轴 -> 世界 -Z，手指竖直向下
夹爪局部 Z 轴 -> 世界 +X，用于侧向推动
```

也就是夹爪从上方竖直下探，然后沿水平方向推动物体。

## 3. 网页展示的是什么

`gui/app.py` 不会重新运行 PyBullet，也不是实时相机页面。

仿真程序会先生成：

```text
results/reveal_push_verify.json
results/reveal_push_verify_viz_data.pkl
```

JSON 中的 `frame_log` 保存每一帧的：

- 物体位置和姿态。
- 夹爪基座位置和姿态。
- 左右手指位置和姿态。
- 当前动作阶段。
- 最终成功或失败状态。

结果文件还会保存 `object_aabb_size`。当默认物体是 URDF 方块、没有 OBJ
mesh 时，网页使用该 AABB 尺寸和逐帧姿态绘制实体方块。

网页读取这些文件后进行逐帧回放：

- OBJ 物体优先绘制真实 mesh。
- 默认 URDF 方块绘制带边框的实体方块。
- 夹爪按照 PyBullet 中的基座和左右手指位姿绘制，但外观使用最早的紧凑样式。
- 点云只作为场景参考，不再用大量蓝色散点代替默认方块。

夹爪分成两层：

```text
PyBullet 物理模型：保持当前稳定夹爪，用于碰撞和推动。
Dash 网页外观：使用最早的小基座 + 两根手指样式，避免长横梁视觉干扰。
```

## 4. 如何验证 `reveal_api.py`

验证应分成两个层次：

1. 验证 `reveal_api.py` 输出的动作计划是否正确。
2. 验证该动作计划交给 PyBullet 后是否真的推动了物体。

### 4.1 动作计划自动测试

运行：

```bash
cd /home/admin128/sangxiyuan/SmartGrasp

conda run -n smartgrasp python execution/test_reveal_api.py
```

测试文件会检查：

- 输入起点 `[0.20, 0.30, 0.04]`。
- Push 向量为 `[0.05, 0.0, 0.0]`。
- 输出目标点为 `[0.25, 0.30, 0.04]`。
- Push 只修改 X，Y 和 Z 保持不变。
- 默认旋转矩阵为竖直 Push 姿态。
- `request_reloop` 为 `True`。
- 错误坐标维度、错误动作类型、零距离和负距离会抛出异常。

当前测试结果：

```text
Ran 2 tests
OK
```

这一步证明 Reveal API 的输入校验、坐标计算和输出字段正确。

### 4.2 PyBullet Push 集成测试

运行：

```bash
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace

conda run -n smartgrasp python scripts/demo_reveal_push.py \
  --distance 0.05 \
  --output results/reveal_push_verify.json
```

该命令会：

1. 创建 PyBullet 桌面和默认小方块。
2. 获取物体中心。
3. 调用 `reveal_api.py` 生成 `+X 0.05 m` Push 计划。
4. 根据物体 AABB 计算接触表面。
5. 驱动夹爪接近并推动物体。
6. 计算物体实际位移。
7. 保存 72 帧左右的 PyBullet 轨迹。

当前验证结果：

```text
请求推动距离：0.0500 m
物体实际 +X 位移：约 0.0480 m
Y 方向误差：约 0.000001 m
Push 成功：True
request_reloop：True
夹爪局部 X 轴：世界 -Z
夹爪局部 Z 轴：世界 +X
```

当前成功判定标准是：

```python
signed_displacement = dot(
    final_position - start_position,
    push_direction,
)
success_threshold = min(0.01, requested_distance * 0.2)
success = signed_displacement >= success_threshold
```

默认请求推动 `0.05 m` 时，物体沿目标方向实际移动至少 `0.01 m` 即判定成功。

## 5. 在网页中检查 Push 轨迹

完成集成测试后运行：

```bash
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace

conda run -n smartgrasp python gui/app.py \
  --host 0.0.0.0 \
  --port 8051 \
  --results results/reveal_push_verify.json \
  --viz-data results/reveal_push_verify_viz_data.pkl
```

浏览器打开：

```text
http://127.0.0.1:8051
```

如果服务运行在远程服务器，需要使用 VS Code 端口转发或 SSH 端口转发。

网页中应重点检查：

- 夹爪是否从物体 X 方向表面外侧接近。
- 夹爪是否与物体发生接触。
- 夹爪是否保持竖直下探姿态。
- 物体是否主要沿 `+X` 移动。
- 物体在 Y、Z 方向是否出现明显异常位移。
- 最后一帧是否显示 `SUCCESS`。

当前默认小方块应显示为带黑色边框的实体方块。如果仍然看到大量蓝色泡泡
遮挡物体，说明浏览器或 Dash 进程仍在使用旧代码。需要重新生成结果、重启
Dash，并在浏览器中按 `Ctrl+Shift+R` 强制刷新。

## 6. 当前验证能证明什么

当前测试能够证明：

- `reveal_api.py` 正确生成固定 `+X`、指定距离的动作计划。
- `reveal_api.py` 输出竖直 Push 夹爪姿态。
- PyBullet 能接收该计划并通过夹爪碰撞推动物体。
- 仿真结果能够记录并在网页端回放。
- `request_reloop` 信号能够写入结果文件。

当前测试不能证明：

- `+X` 是真实遮挡场景中的最佳推动方向。
- 推动后被遮挡目标真正变得可见。
- 动作不会碰撞场景中的其他物体。
- 真实机械臂能够无误差地执行相同动作。
- 仿真质量、摩擦和惯量与真实物体一致。

因此，目前验证的是 Reveal Push 动作生成和单物体 PyBullet 执行链路，而不是
完整的真实机器人遮挡消除闭环。

## 7. 真实 RGB-D 模式的区别

只有运行 `demo_reveal_push.py` 时同时提供以下参数：

```text
--rgb
--depth
--mask
--intrinsics
```

才会使用真实 RGB-D 数据构造仿真物体。

这种模式是：

```text
真实相机 RGB-D 和 Mask
  -> 反投影真实点云
  -> 构造单视角物体凸包
  -> 在 PyBullet 中执行 Push
```

它仍然属于“真实传感器几何 + PyBullet 物理仿真”，不是直接控制真实机械臂。

## 8. 最终推荐运行命令

下面是当前 Reveal Push 仿真的完整推荐流程。

### 8.1 验证 `reveal_api.py`

```bash
cd /home/admin128/sangxiyuan/SmartGrasp

conda run -n smartgrasp python execution/test_reveal_api.py
```

正常结果应包含：

```text
Ran 2 tests
OK
```

### 8.2 运行 PyBullet Push 仿真

```bash
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace

conda run -n smartgrasp python scripts/demo_reveal_push.py \
  --distance 0.05 \
  --mass 0.05 \
  --friction 0.7 \
  --output results/reveal_push_verify.json
```

该命令使用：

- PyBullet 默认小方块。
- 虚拟相机生成的 RGB、Depth 和点云。
- `reveal_api.py` 生成的 `+X 0.05 m` Push 计划。

正常结果应显示：

```text
[Reveal Push] success=True
[Result] .../results/reveal_push_verify.json
[Viz] .../results/reveal_push_verify_viz_data.pkl
```

必须在修改仿真或可视化数据格式后重新执行该命令。当前结果 JSON 会包含：

```text
object_aabb_size
frame_log
actual_displacement
signed_displacement
success
```

其中 `object_aabb_size` 用于在网页中绘制默认 URDF 实体方块。
`frame_log` 中保存的夹爪姿态应满足：局部 X 轴朝世界 `-Z`，局部 Z 轴朝世界
`+X`。

如果要在服务器上同时打开 PyBullet 原生窗口，可增加 `--gui`。通过普通
SSH 且没有图形转发时不要增加该参数。

### 8.3 启动网页回放

在服务器 SSH 终端中运行：

```bash
cd /home/admin128/sangxiyuan/SmartGrasp/graspnet-workspace

conda run -n smartgrasp python gui/app.py \
  --host 0.0.0.0 \
  --port 8051 \
  --results results/reveal_push_verify.json \
  --viz-data results/reveal_push_verify_viz_data.pkl
```

保持该终端运行，不要按 `Ctrl+C`。

启动后，在服务器另一个终端检查：

```bash
curl -I http://127.0.0.1:8051
```

正常结果应包含：

```text
HTTP/1.1 200 OK
```

如果 `curl` 显示连接失败，说明端口转发虽然存在，但服务器上的 Dash
进程没有运行，需要重新执行启动命令。

如果出现：

```text
Address already in use
```

说明 `8051` 已有服务。先运行：

```bash
curl -I http://127.0.0.1:8051
```

如果返回 `HTTP/1.1 200 OK`，直接使用该服务。如果页面仍是旧版本，停止原来
运行 `gui/app.py` 的终端，再重新执行上面的启动命令。

### 8.4 VS Code Remote SSH 直接打开

1. 在 VS Code 底部打开“端口”面板。
2. 选择“转发端口”。
3. 输入远程端口 `8051`。
4. 点击 VS Code 生成的本地地址。

端口转发成功后，本地浏览器直接打开：

```text
http://127.0.0.1:8051
```

浏览器中的 `127.0.0.1` 是本地电脑，因此使用 Remote SSH 时必须存在端口
转发。仅设置 `--host 0.0.0.0` 不能替代 SSH 转发。

当前服务器也可以通过其 VPN 地址访问：

```text
http://100.115.245.13:8051
```

该地址只有在本地电脑能够访问相同 VPN 网络时才有效。

### 8.5 普通 SSH 自动端口转发

如果不使用 VS Code，在本地电脑终端运行：

```bash
ssh -L 8051:127.0.0.1:8051 admin128@labserver0
```

保持 SSH 连接，在本地浏览器打开：

```text
http://127.0.0.1:8051
```

如果本地 `8051` 已被其他程序占用，使用：

```bash
ssh -L 18051:127.0.0.1:8051 admin128@labserver0
```

然后打开：

```text
http://127.0.0.1:18051
```

### 8.6 配置 SSH 后每次直接打开

在本地电脑的 `~/.ssh/config` 中加入：

```sshconfig
Host labserver0
    HostName labserver0
    User admin128
    LocalForward 8051 127.0.0.1:8051
```

以后在本地执行：

```bash
ssh labserver0
```

只要服务器上的 `gui/app.py` 正在监听 `8051`，本地就可以直接打开：

```text
http://127.0.0.1:8051
```

### 8.7 一次运行的终端安排

推荐使用两个服务器终端：

```text
终端 1：运行 demo_reveal_push.py，生成 JSON 和 PKL 后可以退出。
终端 2：运行 gui/app.py，并保持进程持续运行。
```

本地电脑负责建立端口转发并打开浏览器。

### 8.8 页面仍显示旧效果时

按照以下顺序处理：

1. 重新运行 `demo_reveal_push.py`，更新 JSON 和 PKL。
2. 停止旧的 `gui/app.py` 进程。
3. 使用 `8051` 启动新的 `gui/app.py`。
4. 在服务器运行 `curl -I http://127.0.0.1:8051`，确认返回 HTTP 200。
5. 检查 VS Code 的远程端口是 `8051`。
6. 浏览器按 `Ctrl+Shift+R` 强制刷新。

修复后的默认回放应包含：

```text
桌面参考点云
带黑色边框的实体方块
最早紧凑样式的夹爪基座
左手指
右手指
动作阶段文字
```

不应再出现大量蓝色泡泡遮住物体的情况。
夹爪不应再横着伸向物体；它应竖直下探，并用侧向厚度接触物体后沿 `+X`
推动。
