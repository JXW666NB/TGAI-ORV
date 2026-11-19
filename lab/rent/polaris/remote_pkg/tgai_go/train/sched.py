# -*- coding: utf-8 -*-
"""学习率调度器（对齐 torch.optim.lr_scheduler 的 lr 序列）。

设计原则：
  * 只改 `optimizer.lr`，不碰别的状态——所以对任何优化器都生效；
  * `step()` 返回**本次生效的 lr**，同时写回 optimizer，便于训练循环直接用；
  * 公式与 torch 一一对应，测试逐 step 与 torch 对照（见 tests/test_sched.py）。

用法：
    sched = CosineLR(opt, T_max=1000, warmup_steps=50, eta_min=1e-6)
    for step in range(1000):
        ...
        lr = sched.step()          # 先 step 再更新也行，见 docstring
"""
import math

__all__ = ["ConstantLR", "LinearLR", "CosineLR", "StepLR", "LambdaLR", "make_scheduler"]


class _Sched(object):
    """基类：第 0 次 `step()` 返回第 0 个 lr（与 torch 的 get_last_lr 序列一致）。"""

    def __init__(self, optimizer, base_lr=None):
        self.opt = optimizer
        self.base_lr = float(optimizer.lr if base_lr is None else base_lr)
        self._n = 0          # 已 step 次数
        self._lr = None

    # 子类实现：给定"从 0 开始的步号"返回 lr
    def lr_at(self, n):
        raise NotImplementedError

    def step(self):
        lr = self.lr_at(self._n)
        self._n += 1
        self._lr = lr
        self.opt.lr = lr
        return lr

    @property
    def last_lr(self):
        return self._lr

    def state_dict(self):
        return {"n": self._n, "lr": self._lr, "base_lr": self.base_lr}

    def load_state_dict(self, st):
        self._n = int(st["n"])
        self._lr = st["lr"]
        self.base_lr = float(st.get("base_lr", self.base_lr))


class ConstantLR(_Sched):
    """等价 torch.optim.lr_scheduler.ConstantLR(factor)。"""

    def __init__(self, optimizer, factor=1.0, total_iters=None, base_lr=None):
        super().__init__(optimizer, base_lr)
        self.factor = float(factor)
        self.total_iters = None if total_iters is None else int(total_iters)

    def lr_at(self, n):
        # torch: factor 保持 total_iters 个 epoch，之后回到 1.0
        if self.total_iters is None or n < self.total_iters:
            return self.base_lr * self.factor
        return self.base_lr


class LinearLR(_Sched):
    """等价 torch.optim.lr_scheduler.LinearLR（start_factor → end_factor）。"""

    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5,
                 base_lr=None):
        super().__init__(optimizer, base_lr)
        self.start_factor = float(start_factor)
        self.end_factor = float(end_factor)
        self.total_iters = max(1, int(total_iters))

    def lr_at(self, n):
        k = min(int(n), self.total_iters)
        f = self.start_factor + (self.end_factor - self.start_factor) * (k / float(self.total_iters))
        return self.base_lr * f


class StepLR(_Sched):
    """等价 torch.optim.lr_scheduler.StepLR。"""

    def __init__(self, optimizer, step_size, gamma=0.1, base_lr=None):
        super().__init__(optimizer, base_lr)
        if int(step_size) < 1:
            raise ValueError("StepLR 的 step_size 必须 ≥1。修法：传正整数步数")
        self.step_size = int(step_size)
        self.gamma = float(gamma)

    def lr_at(self, n):
        return self.base_lr * (self.gamma ** (int(n) // self.step_size))


class CosineLR(_Sched):
    """余弦退火 + 可选线性 warmup。

    主体等价 torch.optim.lr_scheduler.CosineAnnealingLR(T_max, eta_min)：
        lr(n) = eta_min + (base - eta_min) * (1 + cos(pi * n / T_max)) / 2
    warmup 阶段（前 warmup_steps 步）从 warmup_start*base 线性升到 base，
    之后按上面公式走（warmup 的步数从余弦的 n 里扣除，即 restart 点从 warmup 结束算）。
    """

    def __init__(self, optimizer, T_max, eta_min=0.0, warmup_steps=0, warmup_start=0.0,
                 base_lr=None):
        super().__init__(optimizer, base_lr)
        if int(T_max) < 1:
            raise ValueError("CosineLR 的 T_max 必须 ≥1。修法：传总步数")
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        self.warmup_steps = int(warmup_steps)
        self.warmup_start = float(warmup_start)

    def lr_at(self, n):
        n = int(n)
        if self.warmup_steps > 0 and n < self.warmup_steps:
            f = self.warmup_start + (1.0 - self.warmup_start) * (n / float(self.warmup_steps))
            return self.base_lr * f
        k = max(n - self.warmup_steps, 0)
        # 注意：**不截断**到 T_max —— torch 的 CosineAnnealingLR 在 T_max 之后按 cos 周期回升
        # （warm restart）。实测对比：T_max=10 时第 11 步对应 k=11，lr 会重新变大。
        return self.eta_min + (self.base_lr - self.eta_min) * (1.0 + math.cos(math.pi * k / self.T_max)) / 2.0


class LambdaLR(_Sched):
    """等价 torch.optim.lr_scheduler.LambdaLR（fn 收到步号，返回倍率）。"""

    def __init__(self, optimizer, fn, base_lr=None):
        super().__init__(optimizer, base_lr)
        if not callable(fn):
            raise ValueError("LambdaLR 需要可调用对象。修法：传 fn(step)->倍率")
        self.fn = fn

    def lr_at(self, n):
        return self.base_lr * float(self.fn(int(n)))


def make_scheduler(name, optimizer, **kw):
    """按名字构造。name ∈ {constant, linear, cosine, step, lambda}，大小写不敏感。"""
    table = {"constant": ConstantLR, "linear": LinearLR, "cosine": CosineLR,
             "step": StepLR, "lambda": LambdaLR}
    key = str(name).strip().lower()
    if key not in table:
        raise ValueError("未知调度器 %r。修法：选 %s 之一" % (name, sorted(table)))
    return table[key](optimizer, **kw)
