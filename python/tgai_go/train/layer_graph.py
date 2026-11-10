# -*- coding: utf-8 -*-
"""逐层 CUDA Graph 反向（实验性 API）。

## 它做什么

把每一层 attention 段的「反向（含重算前向）」捕获成一张图；训练时反向阶段逐层重放，
避免逐算子发射的开销。MoE 段仍走逐算子（它有一次主机往返，捕获窗口内非法）。

## 已验证（2026-11-10，V2.7 / V100 / seq=128）

由 `tools/graph_whole_step.py` 实测，同一 harness 对照：

| 项目 | eager | graph |
| --- | --- | --- |
| 前向+loss+反向 | 1843.2 ms | **721.3 ms** |
| loss | 14.45316887 | 14.45316887 |
| 梯度合计 | 1603143.661548 | 1603143.661548 |

相对差 **0.00e+00**（loss 与梯度），加速 **2.56×**。

## 三个必须遵守的前提（都有实测教训）

1. **不要替换 `.grad` 对象**：捕获与重放之间若 `p.grad = None` 或赋新张量，图仍写捕获期缓冲，
   读到的会是新零（曾误报"梯度合计 0.000000"）。本类在捕获后保存 `.grad` 引用，重放后读同一对象。
2. **每步把上游梯度写进固定缓冲**：运行时只有会新建的 `h2d`，所以这里用 `cuMemcpyHtoD`。
3. **输入缓冲固定**：每层一个 `[T, D]` 输入缓冲，每步写入新的层输入。

## 用法

```python
gb = LayerGraphBackward(tr, cos, sin, seq=128)
gb.capture(xs)                      # xs: 每层输入（numpy 列表），一次性
g = gb.replay({0: gw0, 1: gw1, ...})  # 反向：给定每层 dL/dy_attn，返回每层 dL/dx
gb.close()
```
"""
import ctypes

import numpy as np


class LayerGraphBackward(object):
    """逐层 attention 反向的图重放器（实验性）。"""

    def __init__(self, trainer, cos, sin, seq, arena_bytes=256 * 1024 ** 2):
        import tgai_go.train as T
        self.T = T
        self.rt = T._cuda()
        if not hasattr(self.rt, "graph_launch_slot"):
            raise RuntimeError("运行时缺少多图支持" + "。修法：应用 tools/apply_multigraph.py（0.9.11 起已内置）")
        self.tr = trainer
        self.cos = cos
        self.sin = sin
        self.seq = int(seq)
        self.arena_bytes = int(arena_bytes)
        self._x_bufs = []
        self._gw_bufs = []
        self._x_refs = []
        self._grad_refs = []
        self._ready = False

    # ---------- 内部工具 ----------
    def _h2d_into(self, buf, arr):
        a = np.ascontiguousarray(arr, np.float32)
        self.rt._activate()
        self.rt._chk(self.rt.cu.cuMemcpyHtoD(ctypes.c_uint64(int(buf.ptr)),
                                            a.ctypes.data_as(ctypes.c_void_p),
                                            ctypes.c_size_t(a.nbytes)), "cuMemcpyHtoD")

    def _zero_layer(self, i):
        for k in ("q", "k", "v", "o"):
            for m in self.tr.layers[i][k]:
                shp = tuple(int(x) for x in m.weight.shape)
                m.weight.grad = self.T.tensor(np.zeros(shp, np.float32))

    # ---------- 捕获 ----------
    def capture(self, xs):
        """按每层输入捕获图。`xs[i]` 是第 i 层输入（numpy，`[seq, D]`）。"""
        rt, tr = self.rt, self.tr
        NL = len(xs)
        self._x_bufs = [rt.h2d(np.ascontiguousarray(x, np.float32)) for x in xs]
        self._gw_bufs = [rt.h2d(np.zeros((self.seq, tr.D), np.float32)) for _ in range(NL)]
        rt.reserve_capture_arena(self.arena_bytes)
        for i in range(NL):
            self._zero_layer(i)
            if not (rt.graph_arm() and rt.graph_begin()):
                raise RuntimeError("graph_begin 失败（第 %d 层）" % i + "。修法：确认 rt.graph_available() 为真")
            xi = self.T.Tensor._from_gpu(self._x_bufs[i], requires_grad=True)
            ya = tr.fwd_attn(xi, i, self.cos, self.sin)
            (ya * self.T.Tensor._from_gpu(self._gw_bufs[i])).sum().backward()
            if not rt.graph_end():
                raise RuntimeError("graph_end 失败（第 %d 层）" % i + "。修法：看 TG_DEBUG=1 的驱动错误")
            self._x_refs.append(xi)
            self._grad_refs.append([m.weight.grad for k in ("q", "k", "v", "o")
                                    for m in tr.layers[i][k]])
        self._ready = True
        return len(xs)

    # ---------- 重放 ----------
    def replay(self, gws):
        """`gws[i]` = 第 i 层的 `dL/dy_attn`（numpy）。返回每层 `dL/dx_i`（numpy 列表）。"""
        if not self._ready:
            raise RuntimeError("尚未 capture" + "。修法：先调用 capture(xs)")
        out = []
        NL = len(self._x_bufs)
        for i in range(NL - 1, -1, -1):
            self._h2d_into(self._gw_bufs[i], gws[i])
            self.rt.graph_launch_slot(i)
            out.append((i, np.asarray(self._x_refs[i].grad.to_numpy(), np.float32)))
        out.reverse()
        return [g for _i, g in out]

    def replay_layer(self, i, gw):
        """只重放第 i 层：写入该层 `dL/dy_attn`，返回 `dL/dx_i`（numpy）。

        训练循环必须**逐层交错**：MoE 反传得到 `dL/dy_attn` → 本层重放 → 拿到 `dL/dx_i`
        再喂给上一层。所以要这个方法，而不是一次性 replay 全部。
        """
        if not self._ready:
            raise RuntimeError("尚未 capture" + "。修法：先调用 capture(xs)")
        self._h2d_into(self._gw_bufs[i], gw)
        self.rt.graph_launch_slot(i)
        return np.asarray(self._x_refs[i].grad.to_numpy(), np.float32)

    def grads_numpy(self):
        """捕获期 `.grad` 对象当前内容的列表（重放后即该层的参数梯度）。"""
        return [[np.asarray(t.to_numpy(), np.float32) for t in layer] for layer in self._grad_refs]

    def close(self):
        try:
            self.rt.graph_release_all()
        except Exception:
            pass
        self._ready = False
