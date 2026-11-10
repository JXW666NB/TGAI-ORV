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
    def max_affordable(self, n_layers, safety_bytes=2 * 1024 ** 3, fraction=0.65):
        """按**当前设备空闲显存**估算最多能捕获多少张图（每张 `arena_bytes`）。

        实测边界（V2.7 28 层，T=512，每张 480 MiB，模型占 11.3 GiB）：

        | 捕获张数 | arena 合计 | 整步 | 结果 |
        | --- | --- | --- | --- |
        | 4 | 1.9 GiB | 1296 ms | ✅ |
        | 16 | 7.5 GiB | 1165 ms | ✅ |
        | 20 | 9.4 GiB | 1140 ms | ✅ |
        | **24** | **11.3 GiB** | **1058 ms** | ✅ 当前最优 |
        | 26 | 12.2 GiB | 1311 ms | ⚠️ 反而变慢（分配器抖动） |
        | 28 | 13.1 GiB | — | ❌ 读梯度 rc=700 |

        所以不是"越多越好"：超过 ~24 张后剩余显存不足以支撑 MoE 反向的工作集，
        分配器开始抖动，再往上直接非法访存。这里取 `空闲 × fraction − safety`，
        fraction=0.65、safety=2 GiB 在 28 层/T=512 上正好落在 24 张那一档。
        """
        free = self.rt.free_device_bytes()
        if free is None:
            return int(n_layers)
        usable = max(0, int(int(free) * float(fraction)) - int(safety_bytes))
        fit = usable // max(int(self.arena_bytes), 1)
        return int(max(0, min(int(n_layers), fit)))

    def capture(self, xs, max_graphs=None):
        """按每层输入捕获图。

        `xs[i]` 是第 i 层输入（numpy，`[seq, D]`）。`max_graphs=None` 时用
        `max_affordable()` 自动截断（避免抢显存导致重放非法访存）。
        """
        rt, tr = self.rt, self.tr
        NL = len(xs)
        want = int(NL if max_graphs is None else max_graphs)
        if max_graphs is None:
            want = self.max_affordable(NL)
        if want <= 0:
            raise RuntimeError(
                "按当前空闲显存连一张图都放不下（每张需 %.0f MiB）。修法：减小 arena_bytes、"
                "减小 seq/层数，或先 empty_cache()" % (self.arena_bytes / 1024 ** 2))
        self._x_bufs = [rt.h2d(np.ascontiguousarray(x, np.float32)) for x in xs[:want]]
        self._gw_bufs = [rt.h2d(np.zeros((self.seq, tr.D), np.float32)) for _ in range(want)]
        rt.reserve_capture_arena(self.arena_bytes)
        # 捕获失败必须 abort，否则流会一直停在"capture 中"状态，
        # 之后任何 cuCtxSynchronize/empty_cache 都返回 rc=900
        # ("operation not permitted when stream is capturing")，把整台机器后续操作全污染。
        # 这是实测踩到的：arena 用尽一次之后，后续所有尝试都失败在 rc=900。
        for i in range(want):
            self._zero_layer(i)
            begun = False
            try:
                if not (rt.graph_arm() and rt.graph_begin()):
                    raise RuntimeError("graph_begin 失败（第 %d 层）" % i +
                                       "。修法：确认 rt.graph_available() 为真")
                begun = True
                xi = self.T.Tensor._from_gpu(self._x_bufs[i], requires_grad=True)
                ya = tr.fwd_attn(xi, i, self.cos, self.sin)
                (ya * self.T.Tensor._from_gpu(self._gw_bufs[i])).sum().backward()
                if not rt.graph_end():
                    raise RuntimeError("graph_end 失败（第 %d 层）" % i +
                                       "。修法：看 TG_DEBUG=1 的驱动错误")
                begun = False
            except Exception:
                if begun:
                    try:
                        rt.graph_abort()
                    except Exception:
                        pass
                raise
            self._x_refs.append(xi)
            self._grad_refs.append([m.weight.grad for k in ("q", "k", "v", "o")
                                    for m in tr.layers[i][k]])
        self._ready = True
        return len(xs)

    # ---------- 重放 ----------
    @property
    def count(self):
        """实际捕获到的图数量（可能因显存预算少于层数）。"""
        return len(self._x_bufs)

    def has(self, i):
        """第 i 层是否有图可重放；没有就应由调用方走逐算子路径。"""
        return 0 <= int(i) < len(self._x_bufs)

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
