# 训练步 CUDA Graph 化 · 落地方案

> 仓库：`D:\DSH-Workspace\TGAI-对话档案\_tgai_orv`（基线 git HEAD `f302023`；本次**只读**，未改任何被跟踪文件。注：写作期间同批 agent 提交了 `c5b7a6b`（fp16 GEMM 基准 + 更正 Graph 收益预测），我引用到的 §12 行号在该提交后**未变**，见 §0.1）
> 方法：所有结论都能落到文件+行号。标注 **【代码事实】** = 直接读到；**【推断】** = 我基于代码的推理，未实测；**【待实测】** = 我无法在本机验证（本机无 GPU / 环境禁网）。
> 时间：2026-11-02

---

## 0. 前提校正（这一节会改变优先级，请先看）

任务书给的"已知事实"是：**wall 744 ms、GPU 只忙 111.6 ms、利用率 15% ⇒ 瓶颈是 CPU 发射**。

**这条结论在仓库里已经被作者自己作废了。** 【代码事实】

- `docs/GAP_VS_PYTORCH.md:216-251` 是最早那一版：`wall = 744 ms / kernels = 1330 / GPU 忙 = 111.6 ms => 利用率仅 15%`。
- 紧接着 `docs/GAP_VS_PYTORCH.md:261-295` 有标题为「**更正**——之前"GPU 只忙 15%"是我的测量错误」的一节：
  - `:265-267` 认错：`tools/profile_kernel.py` 只 instrument 了 `CudaRuntime.launch()`（NVRTC 内核），**完全没算 cuBLAS 的 385 次 GEMM**，也没算 fast-path 内核；图里实际有 2441 个节点，当时只统计到 1330+385。
  - `:275` 结论：**「原结论作废」**。
  - `:285-290` 已经实测出 Graph 效果：**743 ms → 605.5 ms（1.23×）**，`graph_end=True, nodes=2441`，同形状 PyTorch 610 ms ⇒ 0.99×（追平）。
- git 也印证：`f302023 feat(graph)+docs: 捕获 arena 打通 CUDA Graph（重放 743 -> 605 ms）+ 更正"GPU 只忙 15%"的误判`。
- 顺带一个文档 bug：`tools/profile_kernel.py:7-10` 的 docstring 仍然写着"85% 时间是 CPU 发射开销"，没跟着 §12 更新 —— 这个陈旧字符串就是"15%"继续流传的来源。

**为什么这不只是"数字改一下"，而是方向问题：**

1. 【推断】§12.1 用"两个 event 包住整步"得到 `gpu=734 ms` 其实是**同义反复**：两个事件都在同一条流上，中间的 CPU 空转间隙也被算进这段区间，所以 `gpu == wall` 必然成立，不能证明 GPU 满。
2. 【代码事实】但 §12.2 的**逐算子求和**是可用的下界：cuBLAS 200.7 ms + NVRTC ≥111.6 ms = **≥312 ms**，占 734 ms 的 ≥42%。所以既不是 15%，也不是 100%。
3. 【推断，但很硬】真正能定性的证据是那个 605 ms：图重放已经把**全部** Python/发射开销都消掉了，却仍有 605 ms。如果 GPU 侧只忙 312 ms，重放应该逼近 312~350 ms。既然重放是 605 ms，说明 **≥600 ms 这一步是 GPU 侧真实工作（含 memcpy/memset 节点、未统计的 GEMM）**，CPU 发射不是主体。
4. ⇒ **Graph 能拿到的钱已经基本拿到了（1.23×，已入库）**。再往下压 605→300 要靠减少 GPU 工作（`sum_axis8` 72 次 14.5 ms、softmax_bwd 24 次 8.8 ms、adamw 290 次 18.8 ms 这些），**不是靠继续压发射次数**。

**所以本方案的定位要改一句话：**
> 不是"用 Graph 把 744 打到 150"，而是"**让已经跑通的 605 ms 图，能吃到每步变化的真实数据**"。现在的 605 ms 是在 `ids/tgt` 全程不变的前提下测的（见 §2），**换成真实训练它根本不成立**。这是当前唯一必须解决的 Graph 问题。

【推断】还有一个**方向性机会**，值得单独测：`api.train()` 的默认形状很小（`api.py:84`: `seq_len=256, batch=1`，且 `_make_xy` 产出 `S = seq_len-1`），此形状下 GPU 工作按 `B*S` 缩到 ~1/8（B=4/S=512 → B=1/S=255 是 0.124×），而发射次数（≈1330）几乎不变。**小形状下 Graph 的收益远大于 1.23×**，我估 3~5×（【推断】，必须实测，见 §6.3）。产品默认路径恰好是小形状 —— 这才是 Graph 真正的落点。

### 0.1 与并行工作的交叉影响（写作期间仓库被同批 agent 改动，我未参与）

我工作期间 `docs/GAP_VS_PYTORCH.md` 被另一个 agent 改了并提交为 `c5b7a6b`（新增 §13 + `tools/bench_gemm_fp16.py`，**不是我改的**；我引用的 §11/§12 行号在该提交后未变）。其中两条必须写进本方案：

1. **§11.2 那句"wall 有望从 744 ms 降到 ~150-200 ms"已被明确作废**（改成「CUDA Graph 实测只到 605 ms（1.23×），因为 GPU 本来就是满的」）。这与我在 §0 的独立推断一致 —— 两处独立得出同一结论，可以当作定论。
2. **§13 实测 fp16 GEMM（V100 Tensor Core）5.2~6.2×**，GEMM 占整步 500~600 ms，fp16 后整步估 250~320 ms。
   ⇒ **这直接改变 Graph 的定位**：真正的大头是 GEMM，不是发射。**先上 fp16 GEMM，再上 Graph** 的收益顺序，比反过来好。
   ⇒ 而且两者**会互相抵消**：见下面 §5 R8 —— 捕获期 `_split16_wanted` 主动禁用 split-GEMM，**如果图把 fp16 GEMM 关掉，那 §13 的 5× 就全丢了**。这是本方案必须提前处理的接口问题。
   ⇒ fp16 需要 loss scaling（§13.3 自己写了），**动态 loss scale 的 `found_inf` 回读又是一次 D2H**，和 R1 的 `clip` 是同一个坑，必须一起按"设备侧标量"设计。


---

## 1. 现状盘点（谁存在、谁被调用、缺什么）

### 1.1 Graph 能力：已实现，且**库路径里零调用**

| 函数 | 位置 | 状态 |
|---|---|---|
| `graph_arm()` | `cuda.py:6001-6033` | 建一条非默认流（legacy NULL 流不能捕获），`cublasSetStream_v2` 同步过去 |
| `graph_begin()` | `cuda.py:6035-6067` | 先 `sync()` 排空预热（`:6050`，此时还没开始捕获，合法）；`RELAXED` → `THREAD_LOCAL` 两档降级（`:6057`） |
| `graph_end()` | `cuda.py:6162-6188` | `cuStreamEndCapture` → `cuGraphInstantiate` → 收集 adamw 节点 |
| `graph_launch(bc1,bc2)` | `cuda.py:6190-6213` | 逐节点改 bc1/bc2，再 `cuGraphLaunch` |
| `_graph_destroy()` | `cuda.py:5940-5962` | 销毁 exec/graph，**先**释放 arena（`:5941`，顺序有问题，见 R7） |
| `_collect_adamw()` | `cuda.py:6119-6160` | 按 `func` 指针在 `cuGraphGetNodes` 里认领 adamw 节点，并重建 15 个参数的 ctypes 持有者 |
| `graph_stats()` | `cuda.py:6215-6228` | kernel/blas/launch 计数 + dead/reason/mode |
| `reserve_capture_arena()` | `cuda.py:3023-3037` | 捕获前 `cuMemAlloc` 一大块；`_arena_alloc`（`:3039-3051`）16B 对齐 bump |
| `_alloc()` 捕获分支 | `cuda.py:3102-3113` | `_capturing` 时**禁止** `cuMemAlloc`，只从 arena 切 |

**【代码事实】调用点只有一处：`examples/exam2_chat10m_tgai.py:364-586`。** 全仓 grep `graph_begin|graph_end|graph_launch|reserve_capture_arena`，`python/tgai_go/` 内部**一个调用都没有**。
⇒ **产品路径（`api.train` → `_train_inproc` → `qwen.sft`）与 Graph 完全没接上**。这是"第一步改哪里"的答案基础（见 §8）。

### 1.2 库路径现状（要被 Graph 包住的那条路）

- `api.py:82-86`：`train()` 默认 `seq_len=256, steps=300, lr=5e-6, batch=1, lr_warmup=40, **clip=1.0**`。
- `api.py:207-254` `_train_inproc`：`T.set_default_device(...)` → 建模型 → `Q.sft(...)`（`:243-244`），**没有任何 graph 相关代码**。
- `qwen.py:378-461` `sft()`：真正的训练循环。
  - `:394-403` 用 `np.random.default_rng(seed)` 预生成整条取样计划（shuffle 在 **host**，循环外）。
  - `:412-415` `opt.lr = base_lr * (step+1)/warmup` —— **lr 每步都在变**（warmup 40 步内）。
  - `:424-434` mini-batch 是**逐样本** `forward` + `backward`（`batch=1` 时一次）。
  - `:435` `train_loss = float(loss.to_numpy())` —— **每步一次 D2H + 流同步**。
  - `:438` `O.clip_grad_norm_(model.params(), clip)` —— `clip=1.0` 默认非 0 ⇒ **每步一次 `read_scalar` ⇒ D2H ⇒ 捕获期 poison**（见 R1）。
  - `:441` `opt.step()`。
- `_make_xy`（`qwen.py:331-352`）：`xb = ids[:-1]`、`yb = ids[1:]` ⇒ **`S = seq_len - 1`**（`seq_len=256` 时 S=255，不是 256）。捕获必须按真实 S 捕，不能按 `--seq-len` 的字面值。
- 反传：`__init__.py:1007-1061`，末尾 `rt.sync()`（`:1061`）—— 捕获期被 `sync()` 的守卫吞掉（`cuda.py:2693-2696`），合法但意味着"捕获这一步不建图、也不排空"。
- AdamW：`__init__.py:3961-4019`；`:3981-3996` 每步算 `bc1/bc2/lr/wd_lr` 并 **`_f32()` 收成 float32**；`:4007-4010` 传给 `cuda.adamw(...)`。
- `cuda.adamw()`：`cuda.py:5407-5430`，15 个参数，顺序固定：
  `0 p · 1 g · 2 m · 3 v · 4 n · 5 b1 · 6 omb1 · 7 b2 · 8 omb2 · 9 lr · 10 wd_lr · 11 eps · 12 bc1 · 13 bc2 · 14 do_wd`
  ⇒ 与 `_collect_adamw` 的 `holders[12]/[13]`（`cuda.py:6199-6200`）**对得上**，这是代码事实。

### 1.3 缺失的驱动绑定（`_bind_graph_api`，`cuda.py:5869-5926`）

已绑：`cuStreamBeginCapture(_v2)` / `cuStreamCreate` / `cuStreamEndCapture` / `cuGraphDestroy` / `cuGraphExecDestroy` / `cuGraphLaunch` / `cuGraphGetNodes` / `cuGraphNodeGetType` / `cuGraphKernelNodeGetParams` / `cuGraphExecKernelNodeSetParams` / `cuGraphInstantiateWithFlags`|`cuGraphInstantiate`。

**没绑、但本方案需要：**

| 缺失 API | 用途 | 对应风险/需求 |
|---|---|---|
| `cuGraphExecUpdate` | 整图热更新（形状/拓扑小改时的低成本路径） | §3.1 重捕获的可选优化 |
| `cuGraphExecMemcpyNodeSetParams` + `cuGraphMemcpyNodeGetParams` | 改图内 **H2D 节点**的 src/dst | §2 数据通路（关键） |
| `cuGraphExecMemsetNodeSetParams` + `cuGraphMemsetNodeGetParams` | 改图内 memset 节点 | `full()/zeros()` 捕获时会生成 memset 节点 |
| `cuMemAllocHost` / `cuMemHostRegister` / `cuMemHostUnregister` | **页锁定（pinned）** staging | §2.1（关键，现在完全没有） |
| `cuGraphUpload` | 预上传，消掉首次 `cuGraphLaunch` 的实例化开销 | §5 计时口径 |

---

## 2. Q1：什么东西必须静态？

**总纲（【代码事实】）**：捕获期每次 `launch()` 都会为**每一个节点**新建一套 ctypes 参数（`cuda.py:3303-3321`），其数值被烤进图节点。重放时驱动**只按这些值执行**，Python 一行都不跑。所以：

> 凡是"捕获那一刻取到的值"，重放时都不会自动更新；要么它本来就不该变，要么你必须在每次 `graph_launch` 前显式改写。

### 2.1 逐项判定表

| 项 | 捕获时发生什么 | 重放时能不能更新 | 正确做法 | 类别 |
|---|---|---|---|---|
| **权重 `w`**（`p._gpu`） | 指针烤进 1330 个节点 | **不需要更新**：AdamW 是原地写（`adamw_update` 对 `p` in-place） | 保证指针不变即可。**`.load_state_dict()` 会 `p._gpu = h2d(...)` 换新指针（`__init__.py:2982`）⇒ 图立即失效**，必须先 `_graph_destroy()` | 【代码事实】指针烤死；【推断】换指针=灾难 |
| **梯度 `g`** | `zero_grad()` 把 `p.grad=None`（`__init__.py:2987-2989`），反向时**新分配**缓冲 | **不需要更新**，但**地址必须只属于图** | 捕获期分配的缓冲区在图活着期间**禁止回池复用**（见 R2/R3） | 【代码事实】`:1047-1053` 叶子才挂 `.grad` |
| **AdamW 状态 `m`/`v`** | `AdamW.__init__` 里已 `rt.zeros(...)`（`__init__.py:3969-3979`），**捕获前就存在** | 不需要（原地更新） | 无 | 【代码事实】 |
| **输入 `ids`** | `h2d(..., cache_slot="emb_ids")`（`__init__.py:3061`）→ **捕获期新分配一块设备缓冲**，内容来自 host `hold` | **必须更新**，否则每步都训第一步 | 固定 staging（§2.2） | 【代码事实】`cuda.py:3197-3211` |
| **输入 `tgt`** | `h2d(..., cache_slot="ce_tgt")`（`__init__.py:2822` / `:2907-2908`） | 同上 | 同上 | 【代码事实】 |
| **`loss` 标量** | `ce_fwd` 的输出缓冲，地址烤死 | **重放会写同一个地址**，所以读它就等于读当步 | 保留 Python 引用，重放后 `d2h` 那 4 字节（§3.2） | 【代码事实】`cuda.py:3218-3232` |
| **`lr` / `wd_lr`** | `holders[9]`/`holders[10]`，捕获那一刻的 `_f32(opt.lr)` | ❌ **现在改不了** —— `graph_launch(bc1,bc2)` 只改 12/13（`cuda.py:6199-6200`） | **必须扩参**：`graph_launch(bc1, bc2, lr, wd_lr)` 一并改 9/10；否则 warmup 失效（R4） | 【代码事实】只改 2 个索引 |
| **`bc1`/`bc2`** | `holders[12]/[13]` | ✅ 已支持（`:6195-6204`，`cuGraphExecKernelNodeSetParams`） | 保持；但**每次重放都必须给**（R5） | 【代码事实】 |
| **`opt.t`** | host 侧 Python int，影响 bc 计算 | 纯 host，重放前手动 +1 | 现有示例已这么做（`exam2:515-519`） | 【代码事实】 |
| **shuffle / 采样顺序** | `sft()` 在 `:397-403` **循环外**一次性算完 | ✅ 天然安全（host 侧） | 保持"先算 plan，再逐 step 喂 staging" | 【代码事实】 |
| **dropout** | 全仓 `qwen.py` 无 dropout（grep 无命中） | —— | 无需处理 | 【代码事实】 |
| **dtype/精度模式** | `_split16_wanted`/`_bf16_wanted` 在捕获期**主动禁用**（`cuda.py:4420-4426`、`:4555-4566`） | —— | 见 R8：这会让"图"和"逐算子"用**不同 kernel** | 【代码事实】 |
| **grid/block/shared** | 烤进节点（`_collect_adamw` 的 `grid/block/shared`） | 可改（同一套 `SetParams`），但**形状变了就必须重捕** | 见 §3.1 | 【代码事实】 |

### 2.2 更新指针的正确 API（三条路，按节点类型分）

CUDA 的图重放**不允许**"改一个 Python 变量让图跟着变"。只有三条合法通道：

1. **改节点参数**（kernel 节点）
   `cuGraphExecKernelNodeSetParams(exec, node, &params)` —— 已在用（`cuda.py:6201-6204`）。
   注意：`node` 必须是**原 graph 的节点句柄**（不是 exec 的节点），驱动文档允许这样用；`params.kernelParams` 指向的 host 内存必须在下次调用前保持不变 ⇒ 所以 `_collect_adamw` 把 `holders`/`ptrs` 一起持有（`cuda.py:6141-6160`）是**对的**，不能省。
   *成本*：每节点一次驱动调用，Qwen-0.5B 有 ~218 个参数 ⇒ 218 次/步，估 0.2~0.6 ms/步（【推断】）。所以**不要用它改 lr**，见第 3 条。

2. **改 memcpy / memset 节点参数**
   `cuGraphExecMemcpyNodeSetParams` / `cuGraphExecMemsetNodeSetParams` —— **当前没绑**。
   用途：图内那条 `ids` 的 H2D 节点，如果想换 src 指针，就得走这里。

3. **写设备内存（首选，最便宜）**
   把"每步都会变的标量"放进**一块固定设备缓冲**，kernel 从显存里读，而不是从 kernel 参数里读。每步只需 `cuMemcpyHtoDAsync` 传 16~32 字节，**图节点参数一个都不用动**。
   - `adamw_update` 现在的签名是 15 个标量参数（`cuda.py:968`、`:5421-5430`），要改成 `adamw_update_v2(p,g,m,v,n, const StepScalars* s)`。
   - **这正是 PyTorch 的做法**（见 §7，`torch/optim/adam.py:164-175`：`capturable=True` 时 `step` 常驻设备）。

   【推断】PyTorch 用这条路，不是审美，是因为它没有"改节点参数"的公共 API —— 只有设备侧标量。我们应该照抄，理由同样是**成本**：218 次 `SetParams` vs 1 次 32 字节 H2D。

> **结论（Q1）**：能"捕获后不动"的只有权重、梯度、`m/v`、以及 host 侧的采样计划；**必须每步更新的只有 4 样：`ids`、`tgt`、`bc1/bc2`、`lr/wd_lr`**。前两个走 staging，后两个走"设备标量"（推荐）或节点改写（现状，需扩到 4 个索引）。

---

## 3. Q2：数据通路（`ids`/`tgt` 每步变）

### 3.1 现在为什么必然坏（代码事实）

```python
# cuda.py:3189-3216  h2d()
src  = np.ascontiguousarray(np.asarray(arr), dtype=dtype)   # :3191
buf  = self.empty(src.shape, dtype=dtype)                   # :3197  ← 每步新分配！
hold = np.array(src, copy=True, order="C")                  # :3200  ← 页锁定？不，普通 numpy
cuMemcpyHtoDAsync(buf.ptr, hold.ptr, n, self._stream)       # :3203
buf.host_hold = hold                                        # :3208
if self._capturing:
    self._graph_host_keeps.append(hold)                     # :3209-3211
```

- `cache_slot` 是**按内容**命中的（`:3193` `key=(shape,dtype,tobytes())`）。真实训练里每步 `tobytes()` 都不同 ⇒ **cache 永不命中** ⇒ 每步 `empty()` 走 `_alloc` ⇒ 捕获期只能从 arena 切，重放期地址却还是捕获那一步的。
- 【推断】`hold` 是 `np.array(...)` 的**普通页内存（pageable）**，不是 pinned。CUDA 的流捕获规则里，只有 **pinned** 内存的异步拷贝才能安全地录成 memcpy 节点；pageable 的 `cuMemcpyHtoDAsync` 在驱动内部要先做同步 staging ⇒ 捕获期大概率返回错误或被迫同步。**这是我认为"即使 arena 给够也还是失败"的下一个坑**（【待实测】，本机无 GPU）。

### 3.2 两种做法取舍

#### (a) 固定 staging buffer + `cuMemcpyHtoDAsync`

拆成两层，两层都要有：

- **a-1 设备侧固定**：捕获前 `alloc_static_input("ids", shape, dtype)`，得到**永不复用**的 `GpuBuf`；捕获期 `h2d` 见到 `static_slot` 就直接返回它（不再 `empty()`）。
- **a-2 主机侧 pinned**：`cuMemAllocHost(nbytes)` 拿一块页锁定 staging（**目前没绑，要加**）。
- **捕获方式选一**：
  - **a-A（H2D 录进图，推荐）**：图里保留 `cuMemcpyHtoDAsync(pinned → static_dev)` 的 memcpy 节点。每步只做：`np.copyto(pinned_view, new_ids)`（host memcpy，8 KB 级，<1 µs），然后 `graph_launch`。**每步零额外驱动调用**，数据新鲜。
  - **a-B（H2D 放在图外）**：每步 `cuMemcpyHtoDAsync(static_dev, pinned, n, stream)` 再 `graph_launch`。多 1 次驱动调用（~2 µs），但**不依赖 memcpy 节点能否被捕获** ⇒ **更稳，作为 a-A 的兜底**。

#### (b) 图外做 H2D（= 现状 `h2d` 每步新分配）

**作为"现状"它破坏两件事，不能直接用：**
1. 新分配的地址 ≠ 图里烤的地址 ⇒ 图仍然读旧地址 ⇒ **静默训第一步的数据**（不是报错，是最坏的那种错）。这一点仓库自己也写下来了：`exam2_chat10m_tgai.py:419-421`「Graph 重放时 Python 不执行，新 batch 传不上显存，会一直训第一批」。
2. 捕获期这次分配要 arena 兜，重放期不需要分配但图已经指错地方。

**⇒ (b) 只有在"缓冲区地址固定"的前提下才等价于 (a-B)。"每步新分配"绝对不能进图。**

#### 取舍结论

| | a-A（H2D 进图） | a-B（H2D 图外 + 固定缓冲） | b（现状，每步新分配） |
|---|---|---|---|
| 捕获能过吗 | 【待实测】pageable 会挂；**pinned 才行** | ✅ 一定过（图里根本没有拷贝节点） | ❌ arena 给够也可能挂 |
| 每步额外驱动调用 | 0 | 1 | 1 + 1 次分配 |
| 数据正确性 | ✅ | ✅ | ❌ 静默错 |
| 实现复杂度 | 中（要 pinned + 节点绑定） | **低** | —— |

**建议：先做 a-B（稳、一天能完成、可验收），a-A 作为后续优化。** 两者共用同一套"静态 slot + pinned staging"，切换只是一个开关。

### 3.3 具体接口草案（改哪个函数）

在 `CudaRuntime` 上加（`cuda.py`，紧邻 `reserve_capture_arena`）：

```python
def reserve_static_inputs(self, spec):        # spec: {"ids": (shape, np.int32), "tgt": (shape, np.int32)}
    """捕获前调用。为每个输入建 1 块固定设备缓冲 + 1 块 pinned host 镜像。"""
def static_input(self, name):                  # -> GpuBuf（永不复用）
def fill_static(self, name, arr):              # host: np.copyto 到 pinned 镜像
def push_static(self, name):                   # 设备: cuMemcpyHtoDAsync(pinned -> dev), 排在 self._stream
```

`h2d` 里加一个口子（**这是第一步要改的地方，见 §8**）：

```python
def h2d(self, arr, dtype=np.float32, cache_slot=None, static_slot=None):
    if static_slot:
        buf = self._static[static_slot]        # 捕获/重放都用同一块
        self.fill_static(static_slot, np.asarray(arr, dtype))   # host 侧
        if not self._capturing:                # 图里已有拷贝节点就不重复拷（a-A）；a-B 时照拷
            self.push_static(static_slot)
        return buf
    ...原逻辑...
```

调用侧只需改两处：`__init__.py:3061`（`cache_slot="emb_ids"` → `static_slot="ids"`）和 `__init__.py:2822/2907`（`cache_slot="ce_tgt"` → `static_slot="tgt"`）。

---

## 4. Q3：重捕获策略、池增长、loss 取出

### 4.1 形状变化

- 【代码事实】`graph_begin()` 里如果已有旧图会先 `_graph_destroy()`（`cuda.py:6042-6044`），注释写明「否则池会被冻住」⇒ **支持反复重捕**。
- 【代码事实】但**失败一次就永久死**：`_graph_disable` 设 `_graph_dead=True`（`:5964-5967`），`graph_available()` 之后永远 False（`:5989-5999`）。
- 【推断】所以"形状变了"要分两种：
  - **可预期**（换 seq_len / 换 batch）：**在捕获前就判定**，走"销毁旧图 → 重新 `reserve_static_inputs` + `reserve_capture_arena` → 重捕"。**不要**让形状变化撞进重放路径。
  - **不可预期**（真实数据里偶发变长）：**不要用 Graph**，直接走 eager。判据放在 `sft()` 循环入口，而不是等驱动报错。
- 【推断】重捕代价：`graph_end` 里 `_instantiate_graph` + `_collect_adamw` 要过 2441 个节点（`cuGraphGetNodes` + 每个 `cuGraphKernelNodeGetParams`）⇒ 估 50~300 ms/次。**绝不能每步重捕**，必须是"形状键 → 图"的缓存（key = `(B, S, dtype, 参数集合指纹)`）。

### 4.2 池增长

- 【代码事实】`_pool_frozen()`（`cuda.py:2867-2869`）= `_capturing or _graph_exec`。图活着时 `_pool_shed_ready`（`:2877`）和 `_pool_trim_to`（`:2935`）**都直接 return** ⇒ **池只涨不落**。
- 【代码事实】`empty_cache()`（`:2717-2743`）**没有** `_pool_frozen()` 守卫 ⇒ 它是唯一能在图活着时把块 `cuMemFree` 的出口（R2）。
- 【推断】后果：一次捕获 + 长时间重放，进程的 `reserved` 会顶到峰值不下来；`_raise_alloc_failed` 的提示里已经写了「关掉 TG_CUDA_GRAPH（图捕获会额外占一份显存）」（`:3158`），说明作者也知道。
- **策略**：把"图生命周期"显式化。图活着时：禁止 `empty_cache()`、禁止任何 eager 分配复用图缓冲；需要腾显存时**先 `_graph_destroy()`**（顺带释放 arena），再 `empty_cache()`。

### 4.3 loss 在设备上怎么取出来（这是重点）

**【代码事实】现成的做法已经在跑**：`exam2_chat10m_tgai.py:568` 捕获后把 `loss` 这个 Python 对象存进 `captured_loss`；重放后 `:522` 执行 `captured_loss.to_numpy()`。
`to_numpy` → `d2h`（`__init__.py:1572-1578` → `cuda.py:3218-3232`）→ `cuMemcpyDtoHAsync` + **`cuStreamSynchronize`**（`:3230`）。
因为图的节点地址是固定的，**重放会把当步 loss 写回同一块缓冲**，所以读 `captured_loss` 读到的就是当步值。✅ 这招是对的。

**D2H 同步会不会破坏"低开销"目标？**

【推断】定量算一下就知道**不是问题**：
- 代价 = 一次 `cuStreamSynchronize` = 等当步 GPU 跑完 + 一次驱动往返 ≈ 5~30 µs。
- 这一步 GPU 侧 ≈ 605 ms。**30 µs / 605 ms = 0.005%**。
- 真正的代价不是这 30 µs，而是**失去了"下一步的 CPU 准备"与"当步 GPU"的重叠**。但在 Graph 模式下 CPU 本来就没活干（Python 一行不跑），所以**没有可重叠的东西** ⇒ 同步几乎免费。

**替代方案（如果将来要"每 N 步才取一次"）**：
1. **批量取**：把 N 步的 loss 写进一块 `[N]` 的设备数组（图里加一条 `store` 节点，索引由设备标量给），每 N 步一次 D2H 拿 N 个值。适合 wandb/日志。
2. **异步回读 + 事件查询**：`cuMemcpyDtoHAsync(pinned, ...)` + `cuEventRecord` + 下一步开头 `cuEventQuery`，**永不阻塞**。要求 pinned（现在没有，要加 `cuMemAllocHost`）。
3. **完全不取**：device-side 判据（NaN/Inf 检测、梯度爆炸）也在设备上做，只在异常时回读。
4. **`clip` 的范数同理**：见 R1，必须改成设备侧，否则 `clip=1.0` 的默认路径根本进不了图。

---

## 5. Q4：风险清单（12 条，按严重度排序）

**R1 ·【最高】`clip=1.0` 让库路径必然捕获失败（`nn_ops.py:182-235`）**
`api.train` 默认 `clip=1.0`（`api.py:85`）→ `sft():438` 每步调 `clip_grad_norm_` → `:218` `float(rt.read_scalar(acc))` → `cuda.py:3274-3275` **设 `_graph_poison`**，且 `:3230` 在捕获中的流上做 `cuStreamSynchronize`（非法）→ `graph_end():6172` 直接抛「cuStreamEndCapture ... 捕获中发生 D2H」→ `_graph_disable` **永久回退**。
⇒ **不改这条，产品默认参数下 Graph 100% 不可用。**（示例 `exam2` 能捕获，是因为它压根没做 clip：`exam2:476` 只有 `AdamW(lr=...)`。）
**修法**：`sumsq_parallel` 留在图里 → 设备侧算 `scale = min(1, max_norm/(sqrt(norm²)+1e-6))`（一个 1 线程 kernel）→ `scale_inplace` 从**设备标量**读 scale。全程不回读。

**R2 ·【高】图缓冲被 `empty_cache()` 真正还给驱动 → 重放写已释放显存**
`empty_cache()`（`cuda.py:2717-2743`）不看 `_pool_frozen()`。捕获期释放的临时缓冲（反传里 `grads.pop` 会释放中间梯度，`__init__.py:1057-1060`）会进 `_pool_free`；只要有人图活着时调 `empty_cache()`，这些**被图引用的地址**就 `cuMemFree` 了。**修法**：`empty_cache`/`_pool_trim_to`/`_pool_shed_ready` 全部前置 `_pool_frozen()` 守卫（后两个已有，`empty_cache` 没有）。

**R3 ·【高】图缓冲被池复用给别的张量 → 静默算错**
`_pool_release`（`cuda.py:2820-2865`）**没有**"这块属于 arena/图"的判断，直接 `chain.append(ptr)`。之后任何 `_alloc` 都可能拿到它。PyTorch 专门为这事做了 `check_input_liveness`（`torch/cuda/graphs.py:389-395`），并且明确说**外部 `cuLaunchKernel` 的内核不被追踪** —— 我们正是这种。
**修法**：`_pool_release` 里加 arena 区间判断：
```python
if self._arena_base and self._arena_base <= ptr < self._arena_base + self._arena_size:
    return          # arena 块永不回池（账目由 release_capture_arena 统一结）
```
以及 `_pool_take` 永不返回 arena 地址（上面的改动自动保证）。

**R4 ·【高】`lr`/`wd_lr` 重放时不更新 → warmup 静默失效**
`graph_launch`（`:6195-6204`）只改 `holders[12]/[13]`。`lr` 在 `holders[9]`、`wd_lr` 在 `[10]`。
`sft():412-415` 前 40 步 lr 线性爬升 ⇒ **图里永远是捕获那一步的 lr**，且没有任何报错。
**修法（短期）**：`graph_launch(bc1, bc2, lr=None, wd_lr=None)`，一并改 9/10。**修法（推荐）**：设备标量（§2.2 第 3 条），连 `bc1/bc2` 一起搬，一处 `H2D` 32 字节解决全部。

**R5 ·【中高】不带参数调 `graph_launch()` 会用捕获时的 bc**
`exam2:579` 就是 `graph_launch()`（无参）。那一次侥幸正确（同一 step），但这是个**易踩的坑**。
**修法**：`graph_launch` 里若 `self._adamw_nodes` 非空而 `bc1 is None`，**直接 raise**，不要静默。

**R6 ·【高·正确性】CE 的 `ignore_index` 路径是"数据相关的控制流"，重放会重用捕获那步的 mask**
【代码事实】`__init__.py:2886-2922`：
```python
mask   = (tgt_arr != ignore_index)      # :2889  主机 numpy
nvalid = int(mask.sum())                # :2890  主机标量
rows   = np.flatnonzero(mask)           # :2903  主机索引
ids_buf = rt.h2d(rows,  cache_slot="ce_ids")   # :2904
tgt_buf = rt.h2d(tgt_arr[rows], cache_slot="ce_tgt")  # :2907
lc_buf  = rt.emb_fwd(logits, ids_buf, nvalid, v, n, (nvalid, v))  # :2906 ← grid/形状都来自 nvalid
loss_buf= rt.ce_fwd(lc_buf, tgt_buf, nvalid, v)                    # :2909
```
`nvalid`、`rows`、`emb_fwd` 的 grid 全部由**主机侧的 mask** 决定。SFT 里每条样本长度不同（`_make_xy` 按 `n-1` 掩码，`qwen.py:350-351`）⇒ **`nvalid` 每步都不同** ⇒ 图里烤的是捕获那一步的 `rows`。**重放不会报错，会拿着错误的行集继续训。**
**修法（三选一）**：
1. 图内改为**固定形状**：`nvalid` 恒等于 `n`，把 mask 做成乘 0/1 的权重（多算无效行，但形状稳定）。代码里已经证明"填 safe 目标"会算错 loss（`__init__.py:2883-2884` 注释），所以要用**加权**而不是"填目标"：`loss = Σ w_i·ce_i / Σ w_i`。
2. 把 `rows/nvalid` 留在设备上，grid 用**上界**（`n`）并在 kernel 里 early-return。
3. **（最省事，建议先做）** 捕获期**强制统一 mask 形态**：把 batch 内所有样本补齐到同一 `nvalid`（比如都取 `min` 或都取 `n`）⇒ 拒绝捕获 `nvalid` 会变的 batch。代价是浪费计算，但**正确性优先**。

> 这条是我在整份代码里找到的**最危险的静默错误**。arena 给够、pinned 也做了，只要 CE 这条不修，重放出来的 loss 就是错的。

**R7 ·【中】arena 生命周期 / 释放顺序**
`_graph_destroy()` 在 `:5941` **先** `release_capture_arena()`，`:5949-5960` 才销毁 graph/exec。此刻 exec 还活着且仍引用 arena 地址。虽然紧接着就销毁、不会 launch，但顺序应当反过来：
**`cuGraphExecDestroy` → `cuGraphDestroy` → `release_capture_arena()`**。
另外 `release_capture_arena`（`:3053-3063`）不检查池里是否还有指向 arena 的指针；配合 R3 的修法（arena 块永不回池）才安全。

**R8 ·【中·数值】捕获期会换掉 kernel：`split16` / `bf16` 自我禁用**
`cuda.py:4420-4426`（fp16 split GEMM）和 `:4555-4566`（bf16 GEMM）在 `_capturing` 时**主动返回 False**，注释写明「捕获期间 cast 要新分配」。
⇒ **图里的 GEMM 是 fp32，逐算子执行时是 split-fp16/bf16。** 两者数值不同。
⇒ **只要 `TG_FP16`/`TG_BF16` 开着，"图 vs 逐算子逐位一致"这个验收判据在原理上就不可能成立**（见 §6.2）。必须先约定：验收时把这两个开关关掉，或者接受容差。

> **★ 与并行工作的直接冲突（2026-11-03 新增）**：`docs/GAP_VS_PYTORCH.md` §13 刚实测出 fp16 GEMM 有 5.2~6.2×，是把 744 ms 打到 250~320 ms 的主力。但它和 Graph **在当前实现下互斥**：一进捕获就退回 fp32 GEMM，§13 的收益在图里**全部消失**。
> 也就是说：**只做 Graph 不做 fp16 ⇒ 605 ms；只做 fp16 不做 Graph ⇒ ~250-320 ms；两个都做但不管这个 guard ⇒ 605 ms（fp16 白做）**。
> **必须在 P1 就把 `_split16_wanted`/`_bf16_wanted` 的捕获分支改成"允许"**，配套条件：
> 1. cast 用的 fp16 影子缓冲必须在**捕获前**按固定尺寸预分配（正好是 `reserve_static_inputs` 该管的事）；
> 2. **影子缓存会在图里"冻住"**：`_cast_flat_ptr_bf16`（`cuda.py:4580-4609`）命中缓存时**只返回旧缓冲、不发射 cast kernel**（`:4592-4595`）。如果某个权重的影子在**捕获前就已被热**（预热那步 cast 过），捕获时不会生成 cast 节点 ⇒ 图里**永远没有这条刷新** ⇒ 重放用着**预热的旧权重**去算 GEMM，而 AdamW 每步都在改真权重。这条比 R6 更隐蔽（R6 至少形状对不上会偏，这条数值只差一点、loss 曲线完全正常）。
>    修法：捕获期**强制 `_bf16_shadow` 为空**（不走缓存命中分支），保证每个 cast 都成为图节点；`_bf16_epoch_bump()`（`:4569-4578`）在重放路径上本来就不会被调用（Python 不跑），不再需要。


**R9 ·【中】OOM 时没有回退路径**
`reserve_capture_arena` 失败会**抛** `CudaError`（`:3032-3034`），不是返回 False。谁调它谁得自己兜。
**修法**：包一层 `try/except` → `_graph_disable(ex)` → 继续 eager。**绝不能让"想加速"变成"训练起不来"**。arena 的申请量也要能降级（先试 `need`，失败试 `need/2`，再失败就永久关图）。

**R10 ·【中】`_graph_host_keeps` 只增不减（跨重捕泄漏）**
`graph_begin` 里 `:6053` 重置为 `[]`，但 `_graph_destroy` / `graph_end` **从不清理**。2441 个节点的 ctypes 持有者 + `h2d` 的整块 numpy 副本（`:3211`）会一直挂到下次重捕。重捕 N 次就叠 N 份。
**修法**：`_graph_destroy` 末尾 `self._graph_host_keeps = []`（注意：**必须在销毁 exec 之后**，否则参数内存先没了）。

**R11 ·【中】`_h2d_cache` 与图的所有权冲突**
`_h2d_cache[cache_slot]` 存的是 `GpuBuf`（`:3214-3215`）。若命中，返回的是**上次那块**，可能与图里的地址重叠；`release_capture_arena` 之后 cache 也不失效。
**修法**：图模式下一律走 `static_slot`（§3.3），**绕开 `_h2d_cache`**；并在 `_graph_destroy` 里 `self._h2d_cache.clear()`。

**R12 ·【中】ADAMW 节点"按顺序对齐"的隐含假设（现在无害，将来会炸）**
`_collect_adamw`（`:6137-6143`）用 `zip(found, snaps)` 按 `cuGraphGetNodes` 的返回顺序配对，只校验**数量**相等。目前只改 `bc1/bc2` 且**所有节点这两个值相同**，所以顺序错了也看不出问题。一旦将来按节点改 `lr`（逐参数组不同 lr）或改指针，**顺序就致命**。
**修法**：现在就在 `_note_adamw` 里顺带记 `_ptr(p)`（第一个参数），`_collect_adamw` 时用 `gpu_p` 指针做键配对，而不是顺序。

---

## 6. Q5：验收判据

### 6.1 必须逐位一致，还是允许 1e-6？

**分两档，取决于 R8 那两个开关：**

- **档 A（`TG_FP16=0`、`TG_BF16=0`，纯 fp32）→ 要求逐位一致（0 ULP）。**
  理由（【推断】，但依据充分）：图重放执行的是**同一批 kernel、同一顺序、同一参数、同一条流**。CUDA kernel 对固定 grid/block 是确定性的；cuBLAS 的算法选择只依赖形状/属性，形状不变 ⇒ 选同一个算法、同一块 workspace。所以**没有任何理由产生 1 ULP 的差**。仓库自己也已经实测过一次："首步 loss **逐位相同**（5.925684）"（`docs/PRODUCTION_READINESS.md:307`）。
  ⇒ **1e-6 这个门槛太松了。允许 1e-6 等于放过了"指针指错但数值接近"这类最危险的 bug。**
- **档 B（开了 split16/bf16）→ 逐位不可能（R8），必须容差。**
  建议：`loss` 相对误差 ≤ 2e-6；全参数梯度 `max|Δ|/max|ref|` ≤ 1e-5；`gnorm` 相对 ≤ 1e-6。**并且必须在报告里写明"本档位下图与 eager 走的不是同一套 GEMM"。**

### 6.2 可执行的对比方式（同 seed，N 步，比 loss + 全参数梯度）

```python
# tools/verify_graph_parity.py （新增，不改任何被跟踪文件的行为）
import numpy as np, os, json
import tgai_go.train as T
from tgai_go.train import qwen as Q

N = 20
def run(mode, tag):
    os.environ["TG_CUDA_GRAPH"] = "1" if mode == "graph" else "0"
    T.set_default_device("cuda")
    ck = ...; cfg = Q.cfg_from_checkpoint(ck)
    model = Q.Qwen25(cfg, dict(ck.weights))
    rt = T._cuda()
    # 关键：eager / graph 必须用同一份初始权重（重新从 ck 建模型即可，别复用同一 model）
    losses = Q.sft(model, seqs, steps=N, lr=LR, seq_len=S, clip=0.0,      # ← clip=0，先绕开 R1
                   warmup=0, batch=1, shuffle=True, seed=0, probe=None)
    grads = [g.to_numpy().copy() for g in [p.grad for p in model.params()]]
    params = [p.to_numpy().copy() for p in model.params()]
    np.savez(f"_parity_{tag}.npz", losses=np.array(losses),
             params=np.array(params, dtype=object), grads=np.array(grads, dtype=object))
    return losses, params, grads

a = run("eager", "eager"); b = run("graph", "graph")
assert np.array_equal(a[0], b[0]), "loss 轨迹不是逐位一致"
for i, (pa, pb) in enumerate(zip(a[1], b[1])):
    assert np.array_equal(pa, pb), f"参数 {i} 不是逐位一致"
for i, (ga, gb) in enumerate(zip(a[2], b[2])):
    assert np.array_equal(ga, gb), f"梯度 {i} 不是逐位一致"
```

**要点（每条都对应一个坑）：**
1. **两种模式必须各建一次模型**（从同一 ck 重新加载）。复用同一个 model 会让第二次跑在第一次的终点上。
2. `clip=0.0` 先跑通（绕 R1）；之后再做 `clip=1.0` 的版本。
3. 捕获放在第 3~4 步（给 cuBLAS/RNG 预热），比较**第 5 步之后**的轨迹（示例用 `step == 3`，`exam2:538`）。
4. 必须比 **loss 轨迹 + 全部参数 + 全部梯度**，三者缺一不可：只比 loss 会漏"梯度写错但被平均掉"；只比最终参数会漏"中间某步错"。
5. 失败时**先查 R6（CE mask）**，它是最可能的元凶。

### 6.3 预期 s/step（诚实版）

| 场景 | 现状 | 目标 | 依据 |
|---|---|---|---|
| B=4/S=512, Qwen-0.5B（就是那条 744 ms） | 743 ms eager / **605.5 ms replay（已实测）** | **600~630 ms（1.18~1.24×）** | 【代码事实】`GAP_VS_PYTORCH.md:289`；加上 staging 后每步只多 <1 µs host 拷贝 |
| B=1/S=255（`api.train` 默认量级） | 【待实测】 | 【推断】3~5× | GPU 工作量按 `B*S` 缩到 0.124×（≈39 ms），发射次数不变；图重放后应逼近 GPU 侧 |
| 目标"150 ms" | —— | **在 B=4/S=512 上不可达** | 重放 605 ms 本身就证明 GPU 侧 ≥600 ms（§0） |

**判据（发给用户/上线的门槛）：**
1. 档 A 逐位一致 ⇒ 必须过，否则不上。
2. `median(N=20 步)` 步时：图模式 ≤ 0.85 × eager（否则收益不足以承担复杂度）。
3. 显存峰值不增加超过 1 块 arena（`_arena_off` 实测值，估 0.5~2 GiB）；`L1 显存 ≤ PyTorch 1.1×` 这条既有门槛（`PRODUCTION_READINESS.md:332-339`）不能被打破。
4. 有 `TG_CUDA_GRAPH=0` 的**完全等价回退**，且回退路径开销 ≤ 1%（既有实测 21.7→21.5 ms，1%，`PRODUCTION_READINESS.md:303-304`）。

---

## 7. Q6：PyTorch 的约束，逐条对照

> 以下 PyTorch 结论**不是背的**，是本机 `torch 2.13.0+cpu` 的**源码原文**，行号可查：
> `C:\Users\21233\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\`

| # | PyTorch 的约束（源码证据） | 我们对应什么 | 缺什么 |
|---|---|---|---|
| 1 | **静态输入**：`graphs.py:748-753` —— `if static_input_surface[i].data_ptr() != inputs[i].data_ptr(): static_input_surface[i].copy_(inputs[i])`，然后 `replay()` | `h2d` 每步新分配（`cuda.py:3197`） | ❌ **静态 slot + `copy_` 语义**（§3.3）。这是差距最大的一条 |
| 2 | **静态输出**：`graphs.py:754-758` 把捕获时的张量 `detach()` 直接返回，用户拿到的永远是同一块内存 | `exam2:568` 的 `captured_loss` 已经是这个模式 | ✅ 已等价，做成正式 API 即可 |
| 3 | **静态 grad_output**：`graphs.py:768-774`，同样 `copy_` 进固定缓冲 | 反向根梯度 `rt.full(shape,1.0)`（`__init__.py:1012`）每次新分配，但**捕获后不再被读** | ⚠️ 若将来支持"自定义上游梯度"就要补 |
| 4 | **预热必须在侧流 + 捕获前同步两次**：`graphs.py:646` `torch.cuda.synchronize()` → `:647` `with torch.cuda.stream(torch.cuda.Stream())` 跑 3 次 warmup → `:670` 再 `synchronize()` | `graph_arm` 建非默认流（`cuda.py:6001`）+ `graph_begin` 里 `self.sync()`（`:6050`） | ✅ **语义已对齐**（我们的"侧流"就是那条 `_stream`） |
| 5 | **必须共用同一个 mempool**：`graphs.py:641` `mempool = graph_pool_handle()`，`:672-674`「To avoid replays corrupting each other's memory, capture all passes in the same order they'll run」 | `reserve_capture_arena` 就是我们的 mempool（bump） | ⚠️ 语义有，但**没人调**，且回池路径没封（R2/R3） |
| 6 | **捕获错误模式**：`graphs.py:378-382` `global / thread_local / relaxed`，默认 `global` | `cuda.py:6057` 先 `RELAXED` 再 `THREAD_LOCAL` | ✅ 已对齐（我们更宽松，可接受） |
| 7 | **捕获期禁止任何 CPU-GPU 同步**（`.item()`/`.cpu()`）：`graphs.py:389-395` 的 `check_input_liveness` 就是为抓 use-after-free | `_graph_poison`（`cuda.py:3220/3274`）+ `graph_end` 检查（`:6172`） | ✅ 机制已有；❌ **但 `clip=1.0` 每步都在触发它**（R1） |
| 8 | **参数/缓冲冻结**：`graphs.py:565-574`「only parameters may be trainable. Buffers must have `requires_grad=False`」「may not add or remove any of that Module's parameters」「must not have module hooks」 | `load_state_dict` 会换 `p._gpu` 指针（`__init__.py:2982`） | ❌ **没有校验**。必须在 `graph_begin` 前断言参数指针集合未变 |
| 9 | **AdamW 可捕获 = 状态放设备**：`torch/optim/adam.py:164-175`「Deliberately host `step` on CPU **if both capturable and fused are off**」；`:477-498`（capturable 单张量）、`:733-751`（capturable foreach）里 `bias_correction1 = 1 - beta1**step_t` **全在设备上算**；`:873-875` `lr` 也可以是设备张量 | 我们用 `cuGraphExecKernelNodeSetParams` **改节点参数**（`cuda.py:6201-6204`） | ⚠️ 能用，但**不是 PyTorch 那条路**。PyTorch 的 `capturable=True` 正是我们该抄的：`bc1/bc2/lr` 全放设备标量，1 次 32B H2D 代替 218 次驱动调用 |
| 10 | **AMP/autocast 与捕获冲突**：`graphs.py:580-587`「AMP supported only with disabled caching」「`cache_enabled=False`，否则 raise」 | `_split16_wanted`/`_bf16_wanted` 捕获期禁用（`cuda.py:4420/4555`） | ⚠️ 效果类似（图里走 fp32），但**我们的做法会让图与 eager 数值不同**（R8），PyTorch 是要求你关掉缓存而不是换 kernel |
| 11 | **外部 kernel 不被追踪**：`graphs.py:394-395`「Custom CUDA kernels added outside PyTorch (e.g., via `cuLaunchKernel` or DLPack) are **not** tracked」 | 我们**全部** kernel 都是 `cuLaunchKernel`（`cuda.py:3314/3341/3354`） | ❌ **零保护**。所以 R2/R3 必须靠我们自己封死 |

**一句话总结差距**：PyTorch 用 **"静态张量 + `copy_` + 共享 mempool + 设备侧标量"** 四件事把图变成"可喂数据的黑盒"；我们有 **mempool（arena）和节点改写**，**没有静态输入、没有回池封锁、没有设备侧标量、没有参数冻结校验**。

---

## 8. 落地步骤

### P0（半天，先决策，不写产品代码）

1. **实测复核 §0**：`tools/profile_kernel.py` 把 cuBLAS 也纳入（现在只 instrument `launch()`），并在同一次运行里跑 `TG_CUDA_GRAPH=0/1`。**先把工具 docstring（`:7-10`）那个"85%"改掉**，否则结论会继续被误引。
2. **实测小形状收益**：`--shape 1x256`，eager vs graph。如果这里也只有 1.2×，本方案的价值就只剩"能用真实数据"，要重新排优先级。
3. **量 arena 到底要多少**：捕获一次后打印 `rt._arena_off`（`cuda.py:3037` 起算的 bump 游标）。它就是**该形状下捕获的真实需求**，直接用它定 `reserve_capture_arena` 的值，不要猜。
   - 参考：报错里的 `7340032` = `4 × 512 × 896 × 4`（B=4/S=512/D=896）—— **正好一块 `[B*S, D]` 激活**（【推断】，D=896 来自 `qwen.py:28` 的 `CFG`）。所以"差一块激活"不是"差一点"，而是**每一步都要多备若干块**。

### P1（1~2 天，让图能吃真实数据 = 本方案的核心）

> **第一步改哪里：`python/tgai_go/train/cuda.py` 的 `CudaRuntime.h2d`（`:3189-3216`）。**
> 理由：这是"每步新数据"与"图里固定地址"**唯一正面相撞**的函数。它不动，后面所有工作（arena 调多大、CE 修不修、pin 不 pin）都无法验收。

按顺序：

1. `cuda.py:3023`（紧邻 `reserve_capture_arena`）加 `reserve_static_inputs()` / `static_input()` / `fill_static()` / `push_static()`；`h2d`（`:3189`）增 `static_slot=` 参数（§3.3）。
2. `cuda.py:5869` `_bind_graph_api` 补绑：`cuMemAllocHost`、`cuMemHostRegister/Unregister`、`cuGraphExecMemcpyNodeSetParams`、`cuGraphMemsetNodeGetParams/SetParams`、`cuGraphUpload`。
3. `cuda.py:2820` `_pool_release` 加 arena 区间早退（R3）；`cuda.py:2717` `empty_cache` 加 `_pool_frozen()` 守卫（R2）。
4. `cuda.py:6190` `graph_launch` 扩参 `lr/wd_lr`（R4）+ 无 bc 时 raise（R5）。
5. `cuda.py:5940` `_graph_destroy` 调整释放顺序（R7）+ 清 `_graph_host_keeps`/`_h2d_cache`（R10/R11）。
6. `nn_ops.py:182` `clip_grad_norm_` 改设备侧（R1）—— **这条是"能不能用"的开关，不是优化**。
7. `__init__.py:2886` `_cross_entropy_cuda_ignore` 的 mask 固定化（R6）—— **第一大正确性风险**。
8. `qwen.py:409` `sft()` 循环里加 capture/replay 编排（照抄 `exam2:512-594` 的结构，但把 `xb/yb` 换成 `push_static`）。
9. `api.py:207` `_train_inproc` 加 `graph=True/False` 透传 + **失败自动降级**（R9）。

### P2（可选，纯优化）

- 设备标量结构体 + `adamw_update_v2`，干掉每步 218 次 `SetParams`（对齐 PyTorch `capturable=True`）。
- `cuGraphUpload` 预上传；`a-A` 把 H2D 录进图，省掉每步那次驱动调用。
- 形状键缓存：`(B, S, 参数指纹) → (graph_exec, static_bufs)`。

---

## 9. 我无法验证、必须你们实测的（诚实清单）

1. **pageable 内存的 `cuMemcpyHtoDAsync` 能否被流捕获**（§3.1）。我判断会失败或被迫同步；`cuda.py:3200` 用的是普通 `np.array`，不是 pinned。**这条决定 a-B 还是 a-A。**
2. **捕获期是否真的会生成 memcpy/memset 节点**，以及 `cuGraphExecMemcpyNodeSetParams` 在我们的驱动版本上是否可用（`_instantiate_graph`（`cuda.py:6084-6117`）里已经有"3 参数失败再试 5 参数"的降级先例，照那个模式写）。
3. **`clip=1.0` 是否真的让库路径捕获失败**（R1）。代码路径成立（`read_scalar` → poison → `graph_end` 抛），但我没有 GPU 实测。
4. **605 ms 的测法**：是否含每步 `loss.to_numpy()` 的那次同步（按 `exam2:522` 是含的）。若不含，真实值会略高。
5. **arena 实际需求字节数**（P0-3 的 `_arena_off`）。
6. **小形状（1×256）的 eager/graph 对比** —— 这决定本方案的商业价值。
7. 本机 `torch 2.13.0+cpu` 的源码行号对**其他版本**可能偏移；引用时请以实际版本为准。

---

## 10. 一页速查

- **别再用"15% 利用率"做决策**（`GAP_VS_PYTORCH.md:275` 已作废）。已实测：743 → 605.5 ms，1.23×。
- **图已能跑，卡在"喂不进新数据"**（`exam2:419-421` 自己写了这个限制）。
- **4 个东西必须每步更新**：`ids`、`tgt`、`bc1/bc2`、`lr/wd_lr`。
- **1 个东西绝对不能每步新分配**：任何会成为 kernel 参数的设备缓冲。
- **3 个静默错误源**：CE 的 `nvalid/rows`（R6）、`lr` 烤死（R4）、图缓冲回池（R3）。
- **1 个"根本进不去"的开关**：`clip=1.0`（R1）。
- **验收**：纯 fp32 必须**逐位一致**；开了 split16/bf16 才谈 1e-6（R8）。
- **第一步**：`cuda.py::CudaRuntime.h2d`（`:3189`）。
