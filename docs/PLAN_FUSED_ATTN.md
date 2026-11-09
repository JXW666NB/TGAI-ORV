# 执行计划：把 per-head attention 合并成全宽算子

> 目标：削减每步 15,176 次 kernel 发射（实测：一层前向 208 / 反向 334，×28 层）。
> 现状代码：`python/tgai_go/train/v27_train.py` 的 `V27Trainer.fwd`（第 168–182 行）。

## 现状（实测依据）

```python
h  = L["ln1"](x)                       # 1 次
for hh in range(self.NH):              # 16 个头，每头约 10 次发射
    qh = T._rope_composed(L["q"][hh](h), cos, sin)   # Linear + rope(mul+add)
    kh = T._rope_composed(L["k"][hh](h), cos, sin)
    vh = L["v"][hh](h)
    sc = (qh @ kh.T) * inv            # matmul + mul
    pr = softmax(sc + mask)           # add + softmax
    part = L["o"][hh](pr @ vh)        # matmul + Linear
    acc = acc + part                  # add
```
前向 Top：`ewise_mul_v4` 70、`ewise_add_v4` 70、`softmax_lastdim` 17、`ewise_scale_v4` 16 —— 与 16 头一一对应。

## 改造（分两步，每步都要先写对照）

### 第 1 步：投影合并（风险低、收益确定）

- 在**加载时**把 16 个头的权重拼成全宽 `[D, D]`（q/k/v/o 各一份），存为模型参数：
  `Wq_full = concat([L["q"][h].weight for h in range(NH)], axis=0)`（形状 `[NH*DH, D] = [D, D]`）。
- 前向改为一次 `q = h @ Wq_full.T`，再用 `reshape(T, NH, DH)` 得到各头。
- **反向兼容**：把全宽权重作为**唯一参数**，per-head 参数不再存在；这需要同步改：
  - `params_of(i)` 返回全宽参数（否则优化器拿不到）
  - 检查点加载（`.tg` 里是 per-head 权重）→ 加载时 concat，保存时 split
  - `docs/TRAIN_V27_ORV.md` 里的权重清单/形状说明要更新
- 预期削减：q/k/v/o 从 64 次 Linear 降到 4 次（每层前向少 ~60 次；反向同比）。

### 第 2 步：elementwise 合并（风险中）

- `rope`：把 `cos/sin` 由 `[T, DH]` 平铺成 `[T, D]`（`tile` 一次，代价一次拷贝），
  对全宽 `q` 一次调用 `T._rope_composed` ⇒ 16 次 → 1 次。
- `scale + mask`：全宽 reshape 后对 `[NH, T, T]` 一次性做 `*inv + mask`（需要 batched 算子；
  若后端没有 3D batched matmul/softmax，则 scores/softmax 仍按头循环，只合并 rope 与投影）。
- 预期削减：前向 elementwise 从 140 次降到 ~10 次量级。

## 验收（缺一不可）

1. **同权重对照**：从同一 `.tg` 加载，改造前后对同一输入比较 `logits`，相对误差 ≤1e-6
   （若权重 concat 顺序正确，应当接近逐位相同）。
2. **梯度对照**：同一 `d(loss)/d(logits)` 下比较所有参数的梯度，相对误差 ≤1e-6。
3. **发射计数**：用 `tools/kernel_count.py` 复测一层前向/反向的发射次数，记录前后对比。
4. **单步时间**：用 `tools/full_step_embed.py` 的口径复测；目标把 1.20 s 压到 ≤1.0 s。
5. 全量 `pytest` 全绿；发版 0.9.10（1.0 之前一律 0.9.N）。

## 风险与红线

- **不能**在没做 1/2 对照的情况下直接改前向 —— 之前 MoE softmax 作用域、causal mask 的两次事故
  都是"看起来等价"造成的。
- 权重布局改变会影响**产线引擎**（`tgai_cuda.cu` 侧按 per-head 读），因此：
  **库侧改造必须保持 `.tg` 的 per-head 布局不变**，只在加载后做一次 concat（内存里全宽），
  保存时再 split 回去。这样产线不受影响。
- 若某步实测收益 < 20 ms，就停下，不做第 2 步（避免为小收益承担大风险）。

## 两处自我纠正（2026-11-09，改造前）

1. **我写的对比脚本是无效的**：曾写 `tools/fused_attn_gpu.py`，把"per-head 完整前向"与"只合并了
   投影、根本没做逐头注意力"的版本相比 —— 这个对比没有意义，已删除，不引用它的任何数字。
2. **"ORV 没有切片原语"这句话是错的**：`Tensor` 确实暴露 `__getitem__`（小张量上 `hasattr` 为真）。
   我没验证的是它的**语义与反向**（`grep 'def __getitem__' train/*.py` 只找到 `TGState` 的，
   说明 `Tensor` 的那个是别处定义/挂载的）。

因此第 1 步能否只改投影，取决于 `Tensor.__getitem__` 到底支持什么 —— 这必须**先测**，不能假设。

### 第 0.5 步（新增，任何改造之前）

用一个最小探针：把 `[T, NH*DH]` 张量按 head 切成视图，验证
（a）切片取值与 concat 的逆操作一致；（b）梯度能穿过切片回到原张量。
通过 ⇒ 第 1 步就只是投影改动；不通过 ⇒ 第 1 步实际需要"切片 + batched matmul"两件。

## 重大更正：检查点本来就是全宽（2026-11-09，读 `load_layer` 得到）

`v27_train.py:82-105` 显示：`.tg` 里存的是**全宽**投影权重

```
blocks.%d.attn.q_proj.weight    形状 [D, D]
blocks.%d.attn.k_proj.weight    形状 [D, D]
blocks.%d.attn.v_proj.weight    形状 [D, D]
blocks.%d.attn.out_proj.weight  形状 [D, D]
```

而加载器**主动切成了 per-head**：

```python
for h in range(NH):
    sl = slice(h * DH, (h + 1) * DH)
    L["q"][h].load_state_dict_stream(... ("weight", qw[sl]))
    L["o"][h].load_state_dict_stream(... ("weight", ow[:, sl]))
```

**这推翻了计划里"加载时 concat、保存时 split"的设计**：

- 不需要 concat —— 全宽矩阵已经在 `sd` 里，直接建成一个 `Linear(D, D)` 即可；
- 不需要 split —— 保存时写回 `q_proj.weight` 就是那个全宽矩阵，`.tg` 布局**完全不变**；
- 产线引擎读的是 `.*_proj.weight`，与训练侧表示无关 ⇒ **风险比原估计低得多**。

因此第 1 步简化为：
1. `_build_layer()` 里把 `L["q"]/["k"]/["v"]` 各建成一个 `Linear(D, D)`（`L["o"]` 也是 `Linear(D, D)`），
   而不是 16 个 `Linear(DH, D)`；
2. `load_layer` 直接用 `sd[p + "attn.q_proj.weight"]` 整张装载，不再切片；
3. `fwd` 里用一次 GEMM + 切片取头视图（切片已验证可用）；
4. `params_of` 自动跟着变（`modules_of` 已包含它们，只是列表变短）。

### 最小改造形态（读 `_build_layer` / `modules_of` 后确定，2026-11-09）

`_build_layer`（`v27_train.py:49-54`）把 q/k/v/o 建成**列表**（各 16 个 `Linear`），
而 `modules_of` / `params_of` 依赖列表拼接：

```python
return [L["ln1"], L["ln2"], L["moe"]] + L["q"] + L["k"] + L["v"] + L["o"]
```

**因此最小改造 = 保持"列表"形态，只让列表长度变成 1**：

- `_build_layer`：`fused_heads=True` 时 `L["q"] = [Linear(D, D, bias=False)]`（k/v 同），
  `L["o"] = [Linear(D, D, bias=False)]`；
- `load_layer`：`fused_heads=True` 时整张装载 `sd["blocks.i.attn.q_proj.weight"]`，
  不再切片（`ow[:, sl]` 那两行只在旧路径执行）；
- `fwd`：`fused_heads=True` 时走"一次 GEMM + 切片取头视图"，否则保持现有 16 次循环；
- `__init__` 增加 `fused_heads=False`（**默认关闭**，现有行为逐位不变）；
- `modules_of` / `params_of` **不需要改**（列表长度为 1 时它们自然可用）。

这样默认路径零风险，改造路径可以用一条 GPU 脚本对照验证后再考虑切换默认值。

---

# 附：A2 的实测结果与所需的运行时改动（2026-11-10）

## 已实测（可信，分进程 + CUDA event）

| 项目 | 数值 |
| --- | --- |
| attention 块 eager（seq=128）| 19.32 ms |
| 同上，图重放 | **4.49 ms** |
| 加速 | **4.3×**（另三次复现 2.88× / 3.55× / 3.8×）|
| 梯度等价性 | **q/k/v/o 全部 0.00e+00（逐位相同）** |
| 单层可省 | 14.83 ms ⇒ ×28 层 ≈ 415 ms/步（算术推演）|

## 但**无法直接使用**：运行时只持有一张图

`cuda.py` 里只有 `_graph` / `_graph_exec`，没有图列表（`_graphs`/`_graph_list` 均为 0），
且 `graph_begin` 的注释写明"再捕获一次时丢掉上一张图"。

因此"每层一张图、共 28 张"**当前做不到**。而"整步一张图"又被 MoE 的主机往返阻断
（捕获窗口内禁止 D2H）。

## 需要的最小改动（待授权）

给 `CudaRuntime` 增加**多图支持**，保持现有单图 API 语义不变：

```python
# 新增（不改变现有 _graph/_graph_exec 的用法）
def graph_launch_index(self, i):        # 按索引重放
def graph_count(self):                  # 已捕获多少张
# graph_begin() 追加到列表；graph_launch() 等价于 graph_launch_index(-1 或 0)
```

改动点：`_graphs: list`、`_graph_execs: list`，`graph_end()` 把新图 append 进去（而不是覆盖），
`graph_launch()` 默认重放最近一张；`_graph_destroy()` 释放全部。风险可控：只影响图路径，
非图路径逐算子执行完全不变。

## 拿到授权后的实施顺序

1. 多图支持 + 单元测试（同一进程捕获 2 张图、分别重放、各自结果与 eager 逐位相同）
2. 训练循环：每层 attention 各捕获一张图，MoE 保持 eager；整步 loss 与现路径对照（预期逐位相同）
3. CUDA event 测整步，验证是否落到 ~0.79 s；若达标 → 发 0.9.11



`tests/test_slice_probe.py` 两项**均通过**（CPU）：

```
切片取值 t[:, 1:3] 与 numpy 一致           相对误差 <=1e-6
梯度穿过切片，且准确落在第 1..2 列           相对误差 <=1e-6
```

⇒ **第 1 步可以作为"只改投影"实现**：全宽 GEMM 之后用切片取每个头的 q/k/v 视图，
逐头注意力（scores/softmax/o 投影）保持现状，**不需要 batched matmul**。

