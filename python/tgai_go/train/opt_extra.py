# -*- coding: utf-8 -*-
"""补齐 torch 常用优化器：SGD（含 momentum/nesterov/dampening）、Lion。

实现方式与取舍（重要）：
  * 状态与更新都在**主机端 numpy** 上算，写回参数时：
      - CPU 参数：直接原地写 `p.data[...]`；
      - CUDA 参数：`to_numpy()` 取回、算完 `rt.h2d()` 换 `_gpu` 句柄。
  * 为什么不用 CUDA 内核：这两类更新需要 sign/逐元素组合，当前后端没有对应融合核。
    先用"正确性优先"的实现把语义对齐 torch（有逐 step 对照测试），
    性能优化（融合核）留作后续；大模型日常训练仍建议用 AdamW/Muon 的融合路径。
  * 语义严格对齐：
      SGD：  g = grad + wd*p （wd 为 L2，与 torch 的 weight_decay 一致）
             buf = momentum*buf + (1-dampening)*g
             g   = g + momentum*buf  if nesterov else buf
             p  -= lr*g
      Lion： upd = sign(b1*m + (1-b1)*g) + wd*p
             p  -= lr*upd
             m   = b2*m + (1-b2)*g
"""
import numpy as np

from . import _Optim, _invalidate_fp16_after_step, _bf16_step_boundary, _cuda

__all__ = ["SGD", "Lion"]


def _is_cuda(p):
    return getattr(p, "device", "cpu") == "cuda"


def _read(p):
    if _is_cuda(p):
        return np.array(p.to_numpy(), dtype=np.float32, copy=True)
    return np.array(p.data, dtype=np.float32, copy=True)


def _write(p, arr):
    arr = np.ascontiguousarray(arr, np.float32)
    if _is_cuda(p):
        rt = _cuda()
        p._gpu = rt.h2d(arr)
    else:
        p.data[...] = arr


def _read_grad(p):
    g = p.grad
    if g is None:
        return None
    if _is_cuda(p):
        return np.array(g.to_numpy(), dtype=np.float32, copy=True)
    return np.array(g.data if hasattr(g, "data") else g, dtype=np.float32, copy=True)


class SGD(_Optim):
    """与 torch.optim.SGD 语义一致（momentum / dampening / nesterov / weight_decay）。"""

    def __init__(self, params, lr=1e-3, momentum=0.0, dampening=0.0, weight_decay=0.0,
                 nesterov=False):
        super().__init__(params, lr)
        if float(momentum) < 0:
            raise ValueError("SGD 的 momentum 不能为负。修法：传 >=0 的浮点数")
        if float(weight_decay) < 0:
            raise ValueError("SGD 的 weight_decay 不能为负。修法：传 >=0 的浮点数")
        if nesterov and (float(momentum) <= 0 or float(dampening) != 0):
            raise ValueError("nesterov 需要 momentum>0 且 dampening==0（与 torch 一致）。"
                             "修法：改 momentum 或 dampening")
        self.momentum = float(momentum)
        self.dampening = float(dampening)
        self.wd = float(weight_decay)
        self.nesterov = bool(nesterov)
        self.buf = [np.zeros(p.shape, np.float32) for p in self.params]
        # torch 的 SGD 在**第一步直接把 grad 存进 momentum buffer**（不做 dampening），
        # 从第二步起才乘 (1-dampening)。这一点 torch 文档没强调，但实测如此。
        self._started = [False] * len(self.params)

    @_invalidate_fp16_after_step
    def step(self):
        _bf16_step_boundary()
        lr = float(self.lr)
        for i, p in enumerate(self.params):
            g = _read_grad(p)
            if g is None:
                continue
            pv = _read(p)
            if self.wd:
                g = g + self.wd * pv
            if self.momentum > 0:
                if not self._started[i]:
                    self.buf[i] = g.copy()          # 与 torch 一致：首步 buf = grad
                    self._started[i] = True
                else:
                    b = self.buf[i]
                    b *= self.momentum
                    b += (1.0 - self.dampening) * g
                b = self.buf[i]
                g = g + self.momentum * b if self.nesterov else b
            _write(p, pv - lr * g)


class Lion(_Optim):
    """与论文（Chen et al. 2023）一致；torch 无官方 Lion，测试用手写参考实现对照。"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, lr)
        b1, b2 = float(betas[0]), float(betas[1])
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError("Lion 的 betas 必须在 [0,1)。修法：传 (0.9, 0.99) 之类的值")
        if float(weight_decay) < 0:
            raise ValueError("Lion 的 weight_decay 不能为负。修法：传 >=0 的浮点数")
        self.b1, self.b2 = b1, b2
        self.wd = float(weight_decay)
        self.m = [np.zeros(p.shape, np.float32) for p in self.params]

    @_invalidate_fp16_after_step
    def step(self):
        _bf16_step_boundary()
        lr = float(self.lr)
        for i, p in enumerate(self.params):
            g = _read_grad(p)
            if g is None:
                continue
            pv = _read(p)
            m = self.m[i]
            upd = np.sign(self.b1 * m + (1.0 - self.b1) * g)
            if self.wd:
                upd = upd + self.wd * pv
            _write(p, pv - lr * upd)
            m *= self.b2
            m += (1.0 - self.b2) * g
