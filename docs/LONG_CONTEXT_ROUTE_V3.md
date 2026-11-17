# 长上下文：flash 复用评估（第三版，2026-11-17）

> 结论：**`v3_flash_op.flash_attn` 不能直接用于 CSA 的全局分支**，需要改造成本；
> 而"query 分块 + 重算"可以在不改设备代码的前提下达到同样显存效果。下面是对比。

## 一、`v3_flash_op.flash_attn` 的实际限制（读源码确认）

```python
def flash_attn(q, k, v, causal=True, cq=None, ck=None):
    B, H, S, D = q.shape
    qt = q.reshape(BH, S, D); kt = k.reshape(BH, S, D); vt = v.reshape(BH, S, D)
```

三条硬限制：

| # | 限制 | 影响 |
|---|---|---|
| 1 | **q/k/v 必须同 S**（三处都用同一个 `S` 做 reshape）| CSA 全局分支是 `Lq=T`、`Lk=Tc=T/c` ⇒ **不方阵，直接用会 reshape 报错** |
| 2 | 只接受 `causal` 布尔 | CSA 的掩码是"**chunk 粒度因果**"（`ik <= iq//c`）；seg 时还是**块对角**。布尔表达不了 |
| 3 | 内部 `_fwd_device` 返回 numpy（`o_np`），再 `T.tensor(...)` 拷回显存 | 每次注意力多一次 **D2H+H2D 往返**（16K 时 O = 33 MB 量级）；且反向把 `o_np/lse_np` 常驻在**主机**内存 |

块大小的实测结论（源码注释）：`cq=1024/ck=512` 采用，**块越大越快**（Python 循环 ∝ (S/cq)·(S/ck)，每次启动约 10 µs）。

## 二、两条路线对比

| | A. 改造 flash 设备代码 | B. query 分块 + 重算（纯 Python）|
|---|---|---|
| 改动位置 | `v3_flash_dev._fwd_device/_bwd_device`（设备内核）+ 接口 | `v3_csa._attn` 的调用处 |
| 支持 Lq≠Lk | 要改内核索引 | 天然支持（分块只在 Lq 上）|
| 支持 seg 块对角掩码 | 要在内核里加"文档区间"判定 | 天然支持（掩码按行切片）|
| 显存 | O(S·D) | O(cq·Lk) |
| 计算开销 | 无冗余 | 注意力多一次前向（~1.33× 注意力 FLOPs）|
| 风险 | 高（设备内核 + 索引） | 中（Python，但可逐位对拍）|
| 额外收益 | 快 | 无 |

**建议走 B**：不改设备代码、能覆盖 seg、并能用"与旧路径逐位一致"来验收。

## 三、路线 B 的落地形状（已想清关键点）

`detach()` 在 ORV 上是**拷贝**，所以把 q/k/v 各 detach 一次是 O(T·D) —— **线性**，可接受：

```
qd = q.detach(); kd = k.detach(); vd = v.detach()          # 各 O(T·D)
前向：逐块 qs = q[:, :, i0:i1].detach()                     # 块级小拷贝
      o  = _attn(qs, kd, vd, mask_rows, row_rows, scale)    # 不建图 ⇒ 峰值 O(cq·Lk)
反向：逐块 sub = _attn(qs, kd, vd, ...)                     # 建图，只在块内
      sub.backward(g_chunk)
      收集 qs.grad / kd.grad / vd.grad，累加回 q / k / v
```

关键点：**qs 必须是叶子**（`.detach()` 后），这样 `sub.backward()` 之后能直接读 `qs.grad`，
不会顺着切片反传到 q 的 `_backward` 里去重算整张图。

## 四、验收（与目标一致，且能做得比目标更严）

* 与旧物化路径**逐位对拍**：同一输入下 out / dq / dk / dv 的 `max_abs`（期望 0 或 ~1e-7）
* 显存阶数 p ≤ 1.15；64K 可跑
* `TG_CSA_ATTN=materialized` 可回退

## 五、下一步（本轮不做完，明确交接）

1. 在 `v3_csa.py` 加 `_attn_chunked_ckpt(...)`（约 60 行），只被 `csa_forward` 的**全局分支**调用
2. 用超小模型 `seq=4096/8192` 先对拍（比 64K 便宜），确认逐位一致
3. 再量显存阶数，最后才推 32K/64K
