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


def _auto_compute_dtype():
    """按显卡能力挑**计算**精度：有 bf16 张量核 → bf16；只有 fp16 → fp16；都没有 → fp32。

    存储精度不走这里 —— 见 `T.precision_policy()['storage']`（默认 bf16，任何卡都行）。
    """
    try:
        import tgai_go.train as T
        return T.precision_policy()["compute"]
    except Exception:
        return "fp16"


def _set(on, dtype="auto", weight_cache=True):
    """底层开关：按 dtype 设置 fp16 / bf16 / tf32 的 GEMM 路径。"""
    rt = _rt()
    if not on:
        # 关 AMP 时三条路径都关。否则在 bf16 机器上 amp_off() 只关掉 fp16，
        # bf16 仍在开（默认 dtype 从 'fp16' 改成 'auto' 后这会更常见）。
        f16 = rt._fp16_state() if hasattr(rt, "_fp16_state") else {"on": False}
        f16["on"] = False
        if hasattr(rt, "_bf16_state"):
            rt._bf16_state()["on"] = False
        if hasattr(rt, "f16_cache_enable"):
            rt.f16_cache_enable(False)
        if hasattr(rt, "tf32_enable"):
            rt.tf32_enable(False)
        return f16
    kind = str(dtype).lower()
    if kind in ("", "auto", "none", "bfloat16"):
        # 'auto' 或 bf16 别名都按策略解析；bf16 在没有 bf16 算力的卡上会被降成 fp16
        kind = _auto_compute_dtype() if kind in ("", "auto", "none") else "bf16"
    if kind == "tf32":
        # 只用 cuBLAS 的 TF32 math mode：**不做任何 cast**（0 次发射、0 字节转换），
        # 但拿到的仍是 tensor core。为什么需要这条：`gemm` 模式在主机受限的配置里
        # 被 985 次 cast 发射（11.6 ms/步）拖成负优化，TF32 没有这个成本。
        # 实测 5090/seq512：off 0.1389 → tf32 0.1359（−2.2%），峰值显存不变。
        f16 = rt._fp16_state() if hasattr(rt, "_fp16_state") else {"on": False}
        f16["on"] = False
        if hasattr(rt, "_bf16_state"):
            rt._bf16_state()["on"] = False
        if hasattr(rt, "f16_cache_enable"):
            rt.f16_cache_enable(False)
        if not hasattr(rt, "tf32_enable"):
            raise RuntimeError("该后端没有 TF32 math mode（只有 CUDA 后端有）" +
                               "。修法：用 dtype='fp16'")
        if not rt.tf32_enable(True):
            raise RuntimeError(
                "本机 cuBLAS 不支持 TF32 math mode（sm_70 及更老的卡没有 TF32）" +
                "。修法：改用 dtype='fp16'，或 --amp off")
        return {"on": True, "tf32": True}
    if kind == "fp32":
        # 策略说没有可用的低精度算力：不启用 AMP，保持全 fp32（不是错误）
        f16 = rt._fp16_state() if hasattr(rt, "_fp16_state") else {"on": False}
        f16["on"] = False
        if hasattr(rt, "_bf16_state"):
            rt._bf16_state()["on"] = False
        if hasattr(rt, "f16_cache_enable"):
            rt.f16_cache_enable(False)
        if hasattr(rt, "tf32_enable"):
            rt.tf32_enable(False)
        return f16
    if kind in ("bf16", "bfloat16"):
        if not hasattr(rt, "_bf16_state"):
            raise RuntimeError("该后端没有 bf16 GEMM 支持" + "。修法：用 dtype='fp16'")
        st = rt._bf16_state()
        st["on"] = bool(on)
        # 切换精度时必须把另一条路径关掉，否则 fp16/bf16 会同时开着
        if hasattr(rt, "_fp16_state"):
            rt._fp16_state()["on"] = False
        if hasattr(rt, "tf32_enable"):
            rt.tf32_enable(False)
        return st
    if hasattr(rt, "_fp16_state") is False:
        raise RuntimeError("该后端没有 fp16 GEMM 支持（只有 CUDA 后端有）" +
                           "。修法：T.set_default_device('cuda') 后再用 amp")
    st = rt._fp16_state()
    st["on"] = bool(on)
    if hasattr(rt, "_bf16_state"):
        rt._bf16_state()["on"] = False
    if hasattr(rt, "f16_cache_enable"):
        rt.f16_cache_enable(bool(on) and bool(weight_cache))
    return st


def amp_on(dtype="auto", weight_cache=True):
    """打开 AMP。`dtype='auto'`（默认）按显卡能力自动选计算精度。

    V100（cc 7.0）→ fp16；A100/H100（cc≥8.0）→ bf16；都没有 → 自动保持 fp32。
    **存储**精度另由 `T.precision_policy()['storage']` 决定（默认 bf16，任何卡都支持）。
    """
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
    t32 = rt.tf32_state() if hasattr(rt, "tf32_state") else {"on": False}
    cache = rt.f16_cache_stats() if hasattr(rt, "f16_cache_stats") else {}
    return {"on": bool(f16.get("on")) or bool(b16.get("on")) or bool(t32.get("on")),
            "fp16": bool(f16.get("on")), "bf16": bool(b16.get("on")),
            "tf32": bool(t32.get("on")),
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
        if kind not in ("fp16", "float16", "bf16", "bfloat16", "tf32"):
            raise NotImplementedError("amp 只支持 fp16 / bf16 / tf32（收到 %r）" % (dtype,) +
                                      "。修法：dtype='fp16' / 'bf16' / 'tf32'")
        self.dtype = ("bf16" if kind in ("bf16", "bfloat16")
                      else ("tf32" if kind == "tf32" else "fp16"))
        self.enabled = bool(enabled)
        self.weight_cache = bool(weight_cache)
        self._prev = None

    def __enter__(self):
        self._prev = amp_state()
        _set(self.enabled, self.dtype, self.weight_cache)
        return self

    def __exit__(self, exc_type, exc, tb):
        # 恢复"进入前的状态"：原来开着的 dtype 重新打开，原来关着的就关掉。
        # 注意 `_set(False)` 会把 fp16/bf16/tf32 **三条路径一起关掉**，所以最后
        # 再单独按需恢复 tf32（它与 fp16/bf16 互斥，同一时刻最多一个为真）。
        if self._prev.get("fp16"):
            _set(True, "fp16", True)
        elif self._prev.get("bf16"):
            _set(True, "bf16", True)
        else:
            _set(False, "fp16")
        if self._prev.get("tf32"):
            _set(True, "tf32")
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
