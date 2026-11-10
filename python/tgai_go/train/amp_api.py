# -*- coding: utf-8 -*-
"""可编程 AMP（0.9.20）：`T.amp()` 上下文、`amp_on/amp_off`、`GradScaler`。

## 与 PyTorch `torch.autocast` 的差别（必须说清）

PyTorch 的 autocast 是**逐算子策略**：它把一部分算子（matmul/conv）转 fp16、其它保持 fp32，
激活与梯度因此也可能是 fp16，所以**必须配 GradScaler** 防止梯度下溢。

ORV 的 AMP 走的是**更保守的一条路**：

| 项 | ORV | PyTorch autocast |
| --- | --- | --- |
| 参数（master weights） | **fp32 始终** | fp32（靠 autocast 临时转） |
| 激活/中间量 | **fp32** | fp16（autocast 区域内） |
| GEMM 输入 | fp16（每次调用前 cast） | fp16 |
| GEMM 累加 | **fp32**（`CUBLAS_COMPUTE_32F`） | fp32 |
| 梯度 | **fp32** | fp32（但由 fp16 激活反传，可能下溢） |
| 是否需要 loss scaling | **不需要**（梯度全程 fp32） | 需要 |

所以 `GradScaler` 在这里**不是必需品**；提供它是为了让 PyT 迁移代码不用改，且万一有人把激活也
cast 成 fp16 时可以兜住。默认 `amp()` 不开 scaler。

## 用法

```python
with T.amp():                 # 打开 fp16 GEMM + 权重影子缓存
    loss = model(x)
loss.backward()               # 梯度仍是 fp32，不需要 unscale
opt.step()
```

权重影子缓存会在 `opt.step()` 里自动失效（`Fp16MomAdamW` 已接），自定义优化器请自行
`T.amp_invalidate()`。
"""
import numpy as np

from . import _as_tensor

__all__ = ["amp", "amp_on", "amp_off", "amp_state", "amp_invalidate", "GradScaler"]


def _rt():
    from . import _cuda
    return _cuda()


def _set(on, dtype="fp16", weight_cache=True):
    """底层开关：按 dtype 设置 fp16 或 bf16 的 GEMM 路径。"""
    rt = _rt()
    kind = str(dtype).lower()
    if kind in ("bf16", "bfloat16"):
        if not hasattr(rt, "_bf16_state"):
            raise RuntimeError("该后端没有 bf16 GEMM 支持" + "。修法：用 dtype='fp16'")
        st = rt._bf16_state()
        st["on"] = bool(on)
        return st
    if hasattr(rt, "_fp16_state") is False:
        raise RuntimeError("该后端没有 fp16 GEMM 支持（只有 CUDA 后端有）" +
                           "。修法：T.set_default_device('cuda') 后再用 amp")
    st = rt._fp16_state()
    st["on"] = bool(on)
    if hasattr(rt, "f16_cache_enable"):
        rt.f16_cache_enable(bool(on) and bool(weight_cache))
    return st


def amp_on(dtype="fp16", weight_cache=True):
    """打开 AMP（不用 with 时用这个）。"""
    _set(True, dtype, weight_cache)
    return amp_state()


def amp_off(dtype="fp16"):
    """关闭 AMP（回到全 fp32）。"""
    _set(False, dtype)
    return amp_state()


def amp_state():
    """当前 AMP 状态（给日志/断言用）。没装 CUDA 时返回 `on=False, available=False`，不抛异常。"""
    try:
        rt = _rt()
    except Exception as exc:
        return {"on": False, "available": False, "reason": str(exc)[:120], "cache": {}}
    f16 = rt._fp16_state() if hasattr(rt, "_fp16_state") else {"on": False}
    b16 = rt._bf16_state() if hasattr(rt, "_bf16_state") else {"on": False}
    cache = rt.f16_cache_stats() if hasattr(rt, "f16_cache_stats") else {}
    return {"on": bool(f16.get("on")) or bool(b16.get("on")),
            "fp16": bool(f16.get("on")), "bf16": bool(b16.get("on")),
            "available": True,
            "c32": bool(f16.get("c32", False)), "cache": cache}


def amp_invalidate():
    """参数被外部改写后调用：清空 fp16 权重影子（自定义优化器必须调）。"""
    rt = _rt()
    if hasattr(rt, "f16_cache_clear"):
        return rt.f16_cache_clear()
    return 0


class amp(object):
    """`with T.amp():` —— 打开低精度 GEMM，退出时恢复原状态。

    `dtype`：
      * `"fp16"`（默认）—— V100 也支持；带权重影子缓存。
      * `"bf16"` —— A100/H100 起才算真正加速。**V100 没有 bf16 硬件**：内核仍在，
        但 cuBLAS 会返回错误，运行时按设计**回退 fp32 并只警告一次**（不会算错）。
    """

    def __init__(self, enabled=True, dtype="fp16", weight_cache=True):
        kind = str(dtype).lower()
        if kind not in ("fp16", "float16", "bf16", "bfloat16"):
            raise NotImplementedError("amp 只支持 fp16 / bf16（收到 %r）" % (dtype,) +
                                      "。修法：dtype='fp16' 或 dtype='bf16'")
        self.dtype = "bf16" if kind in ("bf16", "bfloat16") else "fp16"
        self.enabled = bool(enabled)
        self.weight_cache = bool(weight_cache)
        self._prev = None

    def __enter__(self):
        self._prev = amp_state()
        _set(self.enabled, self.dtype, self.weight_cache)
        return self

    def __exit__(self, exc_type, exc, tb):
        # 恢复"进入前的状态"：原来开着的 dtype 重新打开，原来关着的就关掉
        _set(bool(self._prev.get("fp16")), "fp16", True)
        _set(bool(self._prev.get("bf16")), "bf16", True)
        return False


class GradScaler(object):
    """梯度缩放（PyT 迁移兼容用）。

    **ORV 默认 AMP 不需要它**：参数、激活、梯度全程 fp32，只有 GEMM 输入被 cast，
    梯度不会下溢。保留它是为了：
      1. PyT 代码迁移时不用删 scaler 调用；
      2. 用户自己把激活也 cast 成 fp16 时能兜住。
    """

    def __init__(self, init_scale=2.0 ** 16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000):
        self.scale = float(init_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self._good = 0
        self.skipped = 0
        self.steps = 0

    # ---- 缩放 ----
    def scale_loss(self, loss):
        """把 loss 乘上当前 scale（返回新张量，梯度也随之放大）。"""
        return _as_tensor(loss) * self.scale

    # ---- 反缩放 + 检查 ----
    def unscale_(self, params):
        """把梯度除回原尺度；返回是否有非有限值（有则本步应跳过）。"""
        bad = False
        for p in params:
            g = getattr(p, "grad", None)
            if g is None:
                continue
            if hasattr(g, "to_numpy"):
                arr = g.to_numpy()
            else:
                arr = g
            if not np.all(np.isfinite(arr)):
                bad = True
            scaled = arr / self.scale
            try:
                if hasattr(g, "_gpu") and g._gpu is not None:
                    p.grad = _as_tensor(scaled.astype(np.float32)).to("cuda")
                else:
                    p.grad = scaled.astype(np.float32)
            except Exception:
                p.grad = scaled.astype(np.float32)
        return bool(bad)

    def step(self, opt, params=None):
        """等价 PyT 的 `scaler.step(opt)`：先 unscale，非有限则跳过。"""
        if params is None:
            params = getattr(opt, "params", [])
        bad = self.unscale_(params)
        if bad:
            self.scale *= self.backoff_factor
            self._good = 0
            self.skipped += 1
            if hasattr(opt, "zero_grad"):
                opt.zero_grad()
            return False
        opt.step()
        self.steps += 1
        self._good += 1
        if self._good >= self.growth_interval:
            self.scale *= self.growth_factor
            self._good = 0
        return True

    def update(self):
        """PyT 里 `scaler.update()` 的占位（scale 已在 step 内处理）。"""
        return self.scale
