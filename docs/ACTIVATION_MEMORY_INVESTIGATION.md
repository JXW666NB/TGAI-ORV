# 激活显存：两个被源码推翻的假设（2026-11-17）

> 目的：在做"bf16 激活存储"之前先搞清激活显存到底在哪里。
> 下面两条是我先前的判断，**都被源码/实测推翻**。

## 假设一：MoE FFN 的 gate/up 中间量是激活大头 —— ❌ 错

`v3_ops.situ_glu_orv`（L496 起）在 CUDA 上走的是**融合内核**：

```python
if g.device == "cuda":
    # 融合内核：不留中间量（见 situ_gate_orv 的说明）
    return situ_gate_orv(g, u, beta1, beta2) @ w3.transpose(0, 1)
```

注释明确写"不留中间量"，反向由 `situ_glu_bwd` 重算。
⇒ **gate/up 的 (N, d_ff) 中间量根本不驻留**，不能靠量化它们省显存。

## 假设二：`W1[e]` 每次前向都拷贝专家权重（440 MB/步）—— ❌ 错

`v3_moe.py:206` 的注释确实写着"三维张量（`W1[e]` 会拷贝）或**列表**（不拷贝，推荐）"，
但 `v3_train_cli.py:375-377` 的 build **已经用的是列表**：

```python
W1=[p(d_ff, d) for _ in range(E)],
W2=[p(d_ff, d) for _ in range(E)],
W3=[p(d, d_ff) for _ in range(E)],
```

⇒ 列表下标取到的是张量本身，**没有拷贝**。

## 当前掌握的事实

`lab/prof/act_shapes.py`（给 `Tensor._from_gpu` 挂桩，统计前向里创建的
`requires_grad=True` 张量）在 4 层 / seq512 上得到：

| 指标 | 值 |
|---|---:|
| 前向创建的 requires_grad 张量合计 | **1.927 GiB / 585 个** |
| 同一步 `pool.peak_live` | **6.800 GiB** |

**这个口径不足以定位激活**：它统计的是"创建过的"而不是"反向时还活着的"
（视图、以及被释放的中间量都被计进去了），所以 1.93 GiB 与 6.80 GiB 之间的
差额既可能是活跃权重/优化器状态，也可能是我没归因到的驻留激活。

## 结论：下一步必须先建"驻留量"探针，而不是先改 autograd

要做 bf16 激活存储，需要先回答"反向开始那一刻，哪些张量还活着、各占多少"。
方案（下一步实现）：

1. 给 `GpuBuf` 加一个弱引用登记表：`_from_gpu` 登记 `(id, shape, bytes)`，
   `GpuBuf.__del__` 注销；
2. 在 `loss.backward()` 之前打印当前登记表按形状聚合的 top-N。

这比"猜哪个 op 大"可靠；有了它才能决定量化哪些张量、并预测能省多少。
**在拿到这个数据之前不动 autograd。**
