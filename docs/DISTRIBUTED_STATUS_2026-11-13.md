# 分布式现状：修了什么、验证到哪、还没验什么

> 更新：2026-11-13。对应提交见仓库 `main` 分支；本文只记录**本次实测过的**事实，
> 推测一律标注。

## 1. 仓库里现在有两条实现，别混

| 模块 | 机制 | 需要什么 | 状态 |
| --- | --- | --- | --- |
| `python/tgai_go/train/dist.py` | **主机中转**：梯度 D2H → 共享内存（每 rank 一段）→ 汇总 → H2D | 同机多进程即可，**与设备无关**（CPU 也能跑） | **已修 + 已验**（见 §3） |
| `python/tgai_go/train/dist_p2p.py` | **GPU 直连**：CUDA IPC 句柄 + `cuMemcpyPeerAsync`，避开主机往返 | **两张卡**（同机） | **未验证**：本机只有 1 张 V100，`cuMemcpyPeerAsync` 在单卡上没有意义 |

`dist_p2p.py` 注释里"主机往返 0.3~0.5 s，两卡会比一卡慢"是**当初的估计**，本次没有复测。

## 2. 本次修掉的 6 个问题（都是"能跑但错"或"直接卡死"）

按危害排序。每一条都有失败现象 → 根因 → 修法 → 回归用例。

### ① CUDA 广播**静默失效**（最危险）

* **现象**：多卡训练照常跑、loss 照常降，但各卡的**初始权重其实不一样** ⇒ 梯度平均是在
  平均"不同模型"的梯度。
* **根因**：`broadcast()` 读 `p.data`。CUDA 张量的 `.data` 是 `None`（数据在 `._gpu`），而
  `np.ascontiguousarray(None, dtype=np.float32)` **不报错**，返回 `array([nan])`。
  于是每个参数被当成"1 个元素的 nan"：rank0 往共享内存写 nan，其余 rank 把 nan 写进一个
  与张量无关的临时数组。
* **修法**：新增 `_get_data()` / `_set_data()` 按设备取值写回；CUDA 走 `to_numpy()` +
  新增的 `CudaAPI.copy_h2d_into()`（原地写，不新分配）。
* **回归**：`tests/test_dist_device_semantics.py`（含一条把这个 numpy 行为本身钉住的用例）。

### ② 非 0 号 rank 的写回写进了临时数组

* **现象**：即使形状算对，rank≠0 的权重也不会被改写。
* **根因**：`a[...] = ...` 里的 `a` 是 `np.ascontiguousarray(...)` 的结果，不是张量的视图。
* **修法**：同 ①，`_set_data()` 直接写回张量本身。

### ③ 跑得快的 rank 把慢的**等死**

* **现象**：两个 rank 里一个 `rc=0` 正常退出，另一个卡在 `barrier grads_4` 直到超时。
  实测约 **3 次里挂 1 次**。
* **根因**：`close()` 会删掉自己落下的所有 `bar_*.<rank>` 标记；快的一方收尾时删了标记，
  慢的一方还在 `_wait_for` 等那个文件。
* **修法**：`close()` **不再删标记**；残留由下次启动时 rank0 在 `ready` 之前统一清理。
* **回归**：`tests/test_dist_cpu.py::test_missing_grad_on_one_rank_does_not_deadlock`
  （worker 里让 rank0 延迟 0.5 s 进归约，把这个竞态**变成确定性复现**）。

### ④ 某 rank 缺一层梯度 ⇒ 共享内存名字不同 ⇒ 卡死

* **现象**：训练挂死，报错却写"rank0 没创建共享内存"，指向错误方向。
* **根因**：旧实现按"有梯度的条目"拼缓冲。MoE 里某卡没路由到某专家时，两侧条目数不同
  ⇒ 字节总数不同 ⇒ 共享内存名 `tgai_dp_<job>_<need>` 不同 ⇒ 一边创建 A、一边等 B。
* **修法**：按 **`params` 的固定顺序**遍历，缺梯度按 0 补齐（`n` 取自参数本身而不是梯度），
  保证各 rank 字节数一致。
* **回归**：同上。

### ⑤ 缺梯度那一侧的 `grad` 变成 **CPU ndarray**

* **现象**：GPU 探针里 rank1 读 `q.grad.to_numpy()` 报 `AttributeError: 'numpy.ndarray'
  object has no attribute 'to_numpy'`。
* **根因**：`_set_grad()` 按"原来的 grad 是不是张量"判断，缺梯度时 `orig is None`，
  于是走了 `p.grad = val` 兜底 —— 在 CUDA 上留下 CPU 数组。优化器要么报
  "需要 CUDA Tensor 梯度"，要么静默用错。
* **修法**：判定依据改成 **`p.device`**。
* **回归**：GPU 探针断言两个 rank 的 `q.grad` 都是张量且 `device == "cuda"`。

### ⑥ 构造期握手 ⇒ 任何不立刻通信的用法都会卡死

* **现象**：`DataParallel(world=2, rank=1)` 在单测里永久阻塞。
* **根因**：`__init__` 里就等 `ready` 并做首次 barrier。
* **修法**：握手移到惰性 `_start()`，由 `broadcast()` / `all_reduce_grads()` 触发。

## 3. 验证到哪一步（到这里为止是**实测**）

### A. CPU 上两进程（进 CI，`tests/test_dist_cpu.py`）

```
basic   : 广播 [1,1,1] → 两 rank 都是 [1,1,1]；梯度 1 与 3 平均 → [2,2,2]
missing : rank1 缺 q 梯度 ⇒ q 平均 (5+0)/2 = 2.5，且**不再卡死**
```

连续跑 5 次全绿（修 ③ 之前这里 1/3 概率挂）。

### B. **同一张 V100 上跑 2 个 rank**（`tools/dist_gpu_check.py`）

这是本次新增的关键一层：CUDA 路径的问题在 CPU 上根本测不出来。

```
GPU DP 对拍 PASS
  广播后 p=[1.0, 1.0] q=[7.0, 7.0]（两个 rank 一致）
  平均后 p_grad=[2.0, 2.0] q_grad=[2.5, 2.5]（q 在 rank1 无梯度）
```

对拍口径：把"两卡各一份"与"单进程一个大 batch"比较 —— 参数 p 的梯度 (1+3)/2 = 2.0、
参数 q 的梯度 (5+0)/2 = 2.5，与手算一致；同时断言 `q.grad` 在缺梯度的一侧也是 **CUDA 张量**。

### C. 本地套件

`690 passed / 42 skipped`（新增 8 条：设备语义 7 + 两进程缺梯度 1）。

## 4. 还没验证的（别当成都验过）

| 项 | 为什么没验 | 怎么验 |
| --- | --- | --- |
| **真两张卡的加速比 ≥1.7×** | 没有双卡机器 | 租双卡机，跑 `dist.py` 与 `dist_p2p.py` 各一遍，比 step 时间 |
| **`dist_p2p.py` 的任何 GPU 行为** | 单卡上 `cuMemcpyPeerAsync` 无意义 | 同上，双卡机；先验 world=2 的数值对拍再谈性能 |
| 主机中转的**每步开销** | 未测 | 双卡机 profile：D2H+H2D 时间占比 |
| TP / PP | 未实现 | — |
| 检查点并发写 | 未处理（约定只 rank0 写） | 调用方判断 `dp.rank == 0` |

## 5. 怎么跑

```bash
# CPU 两进程正确性（进 CI）
python -m pytest tests/test_dist_cpu.py -q

# 设备语义（不需要卡）
python -m pytest tests/test_dist_device_semantics.py -q

# 同一张卡上 2 个 rank 的 GPU 对拍（需要 CUDA）
python tools/dist_gpu_check.py
```

训练侧接入方式（`dist.py` 文档里有完整契约）：

```python
dp = DataParallel()                       # 从 TGAI_DP_WORLD / TGAI_DP_RANK 读身份
dp.broadcast(params)                      # 训练前一次，保证起点一致
idx = dp.shard_indices(n_samples)         # ★ 各卡必须喂不同数据
...前向/反向...
dp.all_reduce_grads(params)               # 默认求平均 ⇒ 等效 W×B 的大 batch
opt.step()
```

**学习率不会自动改**：默认平均后，等效 batch 变成 `world × per_rank_batch`，
lr 要按这个口径自己调。

## 6. 使用契约（写进 `dist.py` 文档字符串）

1. `broadcast(params)` 训练开始前调一次；
2. `all_reduce_grads(params)` 每个 rank **传完全相同的 params 列表（顺序一致）**；
3. **冻结参数不要传进去**：缺梯度按 0 补齐，而 0 梯度在 AdamW 里仍会走 weight decay；
4. 数据必须各卡不同 —— 用 `shard_indices()`；
5. 检查点只让 `rank == 0` 写。
