# -*- coding: utf-8 -*-
"""补齐 torch 常用优化器与梯度累加簿记：SGD、Lion、Accumulator、scale_grads。

实现方式与取舍（重要）：
  * SGD / Lion 的状态与更新都在**主机端 numpy** 上算，写回参数时：
      - CPU 参数：直接原地写 `p.data[...]`；
      - CUDA 参数：`to_numpy()` 取回、算完 `rt.h2d()` 换 `_gpu` 句柄。
  * 为什么不用 CUDA 内核：这两类更新需要 sign/逐元素组合，当前后端没有对应融合核。
    先用"正确性优先"的实现把语义对齐 torch（有逐 step 对照测试），
    性能优化（融合核）留作后续；大模型日常训练仍建议用 AdamW/Muon 的融合路径。
  * 语义严格对齐：
      SGD：  g = grad + wd*p （wd 为 L2，与 torch 的 weight_decay 一致）
             **首步 buf = g（torch 不做 dampening）**，之后 buf = momentum*buf + (1-dampening)*g
             g   = g + momentum*buf  if nesterov else buf
             p  -= lr*g
      Lion： upd = sign(b1*m + (1-b1)*g) + wd*p
             p  -= lr*upd
             m   = b2*m + (1-b2)*g
"""
import numpy as np

from . import _Optim, _invalidate_fp16_after_step, _bf16_step_boundary, _cuda, _f32

__all__ = ["SGD", "Lion", "Accumulator", "scale_grads"]


def _is_cuda(p):
    return getattr(p, "device", "cpu") == "cuda"


def _cpu_view(buf):
    """把 CPU 侧的存储统一看成 float32 ndarray 视图。

    踩过的坑：`Linear` 的权重 `p.data` 是 **memoryview**，`p.data[...] = x` 会抛
    "memoryview: invalid slice key"；必须先 `np.asarray` 拿到共享内存的 ndarray 再写。
    """
    return np.asarray(buf, dtype=np.float32)


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
        np.copyto(_cpu_view(p.data), arr)


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
    """与论文（Chen et al. 2023）一致；torch 无官方 Lion，测试用独立参考实现对照。"""

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


def scale_grads(params, factor):
    """把所有参数的梯度乘一个标量（梯度累加时通常传 1/accum_steps）。

    CUDA 梯度走 `rt.scale_inplace` 原地做（不产生新缓冲）；CPU 梯度直接 numpy 乘。
    返回真正被缩放的参数个数。
    """
    f = float(factor)
    if f == 1.0:
        return 0
    n = 0
    rt = None
    for p in params:
        g = getattr(p, "grad", None)
        if g is None:
            continue
        if getattr(p, "device", "cpu") == "cuda":
            if rt is None:
                rt = _cuda()
            rt.scale_inplace(g._gpu, _f32(f))
        else:
            buf = _cpu_view(g.data)
            np.copyto(buf, buf * np.float32(f))
        n += 1
    return n


class Accumulator(object):
    """梯度累加簿记（等价于 torch 里手写的 loss/accum_steps + 每 N 步 step）。

    用法：
        acc = Accumulator(model.parameters(), steps=4)
        for batch in loader:
            loss.backward()
            if acc.tick():                 # 累计满 steps 次
                acc.finalize()             # 默认按 1/steps 平均（average=True）
                clip_grad_norm_(params, max_norm)   # 裁剪在累加**之后**生效
                opt.step(); opt.zero_grad(); acc.reset()

    为什么需要 `tick()` 返回布尔：ORV 没有 torch 那种"每步自动清零"的语义，
    没有簿记很容易忘记 scale/step，导致"等效 batch 变大但梯度没平均"。
    """

    def __init__(self, params, steps, average=True):
        self.params = list(params)
        self.steps = int(steps)
        if self.steps < 1:
            raise ValueError("Accumulator 的 steps 必须 >=1。修法：传正整数微批次数")
        self.average = bool(average)
        self._n = 0
        self._scaled = False

    def tick(self):
        """记一次微批。返回 True 表示已累计满，应当 step。"""
        self._n += 1
        return self._n >= self.steps

    def finalize(self):
        """按 1/steps 缩放梯度（average=True 时），返回被缩放的参数个数。"""
        if self._scaled or not self.average:
            return 0
        self._scaled = True
        return scale_grads(self.params, 1.0 / float(self.steps))

    def reset(self):
        self._n = 0
        self._scaled = False

    @property
    def pending(self):
        return self._n
