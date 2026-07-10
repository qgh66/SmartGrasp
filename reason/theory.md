# 推理理论说明

本文档用中文整理当前 `reason` 模块中 `partially_occluded` 和
`fully_occluded` 两个分支的打分逻辑，并给出一个可以后续实现的
“体积感知概率模型”。

这里把“partial-occupied / fully-occupied”对应到代码中的
`partially_occluded / fully_occluded` 分支。

## 阅读依据

- Tang 等，2025，AffordGrasp 论文。
  核心思想：视觉语言模型可以把语言和图像分解成任务、物体、可抓取部件和 affordance，
  形式上可写为 `T, O, p*, a* = GPT-4o(L, I)`。
- Qian 等，2026，ThinkGrasp 论文。
  核心思想：高层目标选择可以写成关于语言、场景上下文和候选物体的最大化问题，
  同时考虑任务相关性、抓取难度和遮挡关系。
- Jiao 等，2025，FreeGrasp 论文。
  核心思想：遮挡图可以提供动作序列。先剪掉遮挡面积比例过小的边，再从目标节点向
  顶层遮挡物遍历，就能得到第一个应该抓取或移除的物体。
- Bejjani 等，2021，遮挡感知物体检索论文。
  核心思想：遮挡场景中的隐藏目标搜索天然是一个信念空间问题；每个动作都会更新
  对可能状态的概率分布。
- Lei 等，2026，ActiveGrasp 论文。
  核心思想：面向抓取任务的信息增益可以定义为某个任务相关分布的熵下降，
  而不只是普通的可见性分数。
- Breyer 等，2021，Volumetric Grasping Network 论文。
  核心思想：体素化表示可以承载抓取质量、姿态和夹爪宽度等信息，因此后续可以把当前
  的二维半体积代理替换成真正的三维体积表示。

## 当前变量

记：

- `G = (V, E)` 为有向遮挡图。
- 边 `u -> v` 表示物体 `u` 遮挡物体 `v`。
- `r_uv in [0, 1]` 是边上的接触比例，在代码中保存为 `ratio`。
- `t` 是目标物体编号。
- `C` 是当前可直接抓取的顶层候选集合。
- `s_i` 是视觉语言模型给候选 `i` 的语义分数或概率。
- `g_i` 是候选 `i` 的几何分数。
- `q_i` 是候选 `i` 的可抓取性分数。若有部件级分数，通常取
  `max_part q_{i,part}`。
- `P(i)` 是排序模块使用的归一化信念。
- `H(P) = - sum_i P(i) log2 P(i)` 是以 bit 为单位的香农熵。

共享的融合形式可以看作乘积专家模型：

```text
raw_i = s_i^beta * g_i^gamma
P(i) = raw_i / sum_j raw_j
```

当前实现中 `beta = gamma = 1`。

这个形式是合理的：视觉语言模型提供语义和任务相关性，图结构、深度和 mask 提供物理
可行性。把两者相乘再归一化，相当于把两个不同来源的证据融合成候选物体上的决策信念。

## 部分遮挡目标

目标已经出现在遮挡图中，但存在祖先节点。当前动作只能移除顶层祖先：

```text
A_t = ancestors_G(t)
C_t = { i in A_t : in_degree_G(i) = 0 }
```

### 语义先验

`reason/partially_visible/prior.py` 会让视觉语言模型对目标的所有祖先节点打分。
这些分数是独立分数，不是互斥概率：

```text
s_i ~= p(候选 i 对目标遮挡链重要 | L, I, G)
```

当前动作只使用顶层候选的分数，但较低层祖先的分数会被缓存起来，用于移除某个物体后
重新计算反事实熵。

### 几何先验

当前几何分数是所有路径的边比例乘积之和：

```text
g_i = sum_{路径 pi: i -> t} product_{(u,v) in pi} r_uv
```

解释：如果把每条边的 `ratio` 理解为一个物体实质性遮挡下一个物体的强度或概率，
那么一条路径上的乘积就是这条遮挡链的强度。对所有路径求和，可以近似得到候选 `i`
遮挡目标 `t` 的总图结构支持。

从数学表达上，一个更干净的有界版本是：

```text
path_strength(pi) = product_{(u,v) in pi} r_uv
g_i = 1 - product_{路径 pi: i -> t} (1 - path_strength(pi))
```

当前的求和形式可以看作这个并集概率的一阶近似。由于后续还会归一化，当前版本可以用；
但有界版本不会超过 1，在论文中更容易解释。

### 信念和熵

对当前顶层候选集合：

```text
P_0(i) = normalize_i(s_i * g_i), i in C_t
H_0 = H(P_0)
```

这里的熵表示“下一步应该移除哪个顶层遮挡瓶颈”的不确定性。熵越小，说明策略对下一个
要移除的候选越确定。

### 反事实结构信息增益

若动作 `a` 表示移除候选 `a`，构造残余遮挡图：

```text
G_-a = 移除节点 a 后的 G
C_-a = G_-a 中目标 t 的顶层祖先集合
```

如果已经没有候选，说明目标可以直接抓取：

```text
H_-a = 0
```

否则，在新的候选集合上重新构建后验，并复用已经缓存的视觉语言模型祖先分数：

```text
P_-a(i) = normalize_i(s_i * g_i(G_-a)), i in C_-a
H_-a = H(P_-a)
IG_partial(a) = H_0 - H_-a
```

这可以解释为“结构熵下降”。它不是严格意义上同一个随机变量前后的互信息，
因为移除物体后候选集合可能发生变化。更稳妥的解释是：

```text
IG_partial(a) = 当前子问题的不确定性
                - 移除 a 后残余子问题的不确定性
```

当前旧版效用函数是：

```text
U_legacy(a) = P_0(a) * IG_partial(a) - alpha * cost(a)
```

当前实验中的两个变体是：

```text
U_ig(a) = IG_partial(a)
U_ig_graspability(a) = IG_partial(a) * q_a
```

更适合论文表述的形式是：

```text
IGn(a) = max(0, IG_partial(a)) / log2(max(2, |C_t|))
U_partial(a) = q_a * P_0(a) * IGn(a) - lambda * cost(a)
```

这里把负的信息增益截断为 0，是因为如果移除某个物体会增加后续不确定性，
它就不应该从信息项里得到奖励。

## 完全遮挡目标

目标没有出现在可见遮挡图中，因此可以把隐藏目标所在位置建模成一个潜变量：

```text
Z in C
C = { 可见顶层物体 }
```

此时视觉语言模型输出的是互斥概率：

```text
s_i ~= p(Z = i | L, I)
sum_i s_i = 1
```

### 当前体积代理

`reason/invisible/geometry.py` 会给每个可见候选计算一个二维半体积代理：

```text
A_i = 物体 i 的可见 mask 面积
h_i = max(1, 桌面深度 - 物体 i 的平均深度)
```

如果有物体压在候选 `i` 上方，代码使用等效面积：

```text
Aeq_i = (A_i + sum_{k in predecessors(i)} A_k) / (1 + num_predecessors(i))
Vproxy_i = Aeq_i * h_i
```

然后归一化为体积先验：

```text
v_i = Vproxy_i / sum_j Vproxy_j
```

这不是真正的物理体积，而是“可隐藏空间容量”的代理。直观上，可见面积越大、
高度信号越明显，越可能对应一个可以藏住目标的遮挡区域。

融合后的隐藏目标信念为：

```text
P_0(i) = normalize_i(s_i^beta * v_i^gamma)
H_0 = H(P_0)
```

### 期望信息增益

对动作 `a`，观测结果有两种：

```text
命中：目标在 a 后方或下方被找到，概率为 P_0(a)
未命中：目标不在那里，概率为 1 - P_0(a)
```

如果命中，剩余熵为 0。如果未命中，则需要更新剩余候选上的信念。
纯贝叶斯形式的未命中后验为：

```text
P_miss(j) = P_0(j) / (1 - P_0(a)), j != a
```

当前代码选择在移除 `a` 后重新计算未命中分支的信念，这也是合理的，因为移除动作会改变
场景上下文：

```text
P_miss = posterior(C_-a | 移除 a 且未找到目标)
H_miss = H(P_miss)
IG_full(a) = H_0 - (1 - P_0(a)) * H_miss
```

这是标准的期望熵下降目标。它比部分遮挡分支的“结构信息增益”更严格，
因为这里始终围绕同一个潜变量 `Z`，并且显式边缘化了命中和未命中两种结果。

当前效用函数是：

```text
U_full(a) = IG_full(a)
U_full_graspability(a) = IG_full(a) * q_a
```

更适合论文表述的形式是：

```text
IGn(a) = IG_full(a) / log2(max(2, |C|))
U_full(a) = q_a * IGn(a) - lambda * cost(a)
```

默认情况下不要再额外乘一次 `P_0(a)`，因为 `P_0(a)` 已经通过命中概率进入了
`IG_full(a)`。

## 体积加视觉语言模型概率

你提出的“结合视觉语言模型和使用体积等给出的概率 p 作为原始概率，算出应该抓取哪个
物体”，可以写成：

```text
p_vlm(i) = 归一化后的视觉语言模型概率，或归一化后的独立语义分数
p_vol(i) = V_i / sum_j V_j
p_graph(i) = 图路径概率，如果当前分支可用

p0(i) = normalize_i(
    p_vlm(i)^beta *
    p_vol(i)^gamma *
    p_graph(i)^eta
)
```

对完全遮挡目标：

```text
V_i = Vproxy_i = Aeq_i * h_i
分数(i) = q_i * EIG(i; p0) - lambda * cost(i)
```

对部分遮挡目标：

```text
V_i = A_i * h_i                         # 可选的局部可移除性先验
p_graph(i) = i 到目标的有界路径概率
分数(i) = q_i * p0(i) * IGn(i) - lambda * cost(i)
```

如果后续有真正的三维重建，可以把当前二维半体积代理替换为体素或点云体积：

```text
V_i = sum_{x in Omega_i} voxel_volume * 1[x 是未知区域或可隐藏目标区域]
```

或者替换为带抓取质量的体素表示：

```text
V_i = sum_{x in Omega_i} voxel_volume * hideability(x) * grasp_quality(x)
```

然后仍然使用：

```text
p_vol(i) = V_i / sum_j V_j
```

这样可以保持同一套概率流程，同时提升体积项的物理含义。

## 实践建议

1. `fully_occluded` 分支应保留香农熵和期望信息增益。它的推导最干净：
   隐藏目标位置是潜变量，每个移除动作对应命中或未命中两种结果。
2. `partially_occluded` 分支也可以保留熵，但写作时建议称为“结构信息增益”
   或“反事实熵下降”。它比较的是当前子问题和残余子问题的不确定性，
   不宜直接声称为严格互信息。
3. 如果要写进最终算法，建议把部分遮挡分支中的路径求和几何项替换成有界并集形式，
   或者新增为一个可选 ranking 模式。
4. 当前完全遮挡分支中的 `area * height` 应诚实地称为 `Vproxy`，不要称为真实体积。
   它可以作为隐藏空间容量先验使用。
5. 建议保留 `beta`、`gamma` 和 `eta` 这类指数权重，用来校准语义、几何和体积项。
   视觉语言模型分数默认并不是严格校准概率。
6. 推荐使用分支特定的效用函数：
   - 部分遮挡：`q_i * p0(i) * IGn(i) - lambda * cost(i)`
   - 完全遮挡：`q_i * EIG(i; p0) - lambda * cost(i)`

## 最小算法草图

```text
输入：perception，目标 t，分支 b
输出：下一步应该抓取或移除的物体

1. 构造候选集合 C。
2. 获取 C 上的视觉语言模型分数 p_vlm。
3. 当目标出现在遮挡图中时，计算图结构先验 p_graph。
4. 当深度和 mask 可用时，计算体积先验 p_vol。
5. 融合得到 p0 = normalize(p_vlm^beta * p_graph^eta * p_vol^gamma)。
6. 对每个候选 a：
   如果 b 是 partially_occluded：
      计算残余图 G_-a。
      计算 IG_struct(a) = H(p0) - H(残余候选上的后验)。
      分数(a) = graspability(a) * p0(a) * IGn(a) - lambda * cost(a)。
   如果 b 是 fully_occluded：
      计算移除 a 后的未命中后验。
      计算 EIG(a) = H(p0) - (1 - p0(a)) * H_miss(a)。
      分数(a) = graspability(a) * EIGn(a) - lambda * cost(a)。
7. 返回分数最高的候选。
```
