# -*- coding: utf-8 -*-
"""高层偏好/策略优化训练器：`DPOTrainer` 与 `GRPOTrainer`。

## 为什么这样设计

`align` 里已经有与 torch 逐位对照通过的**损失件**（`dpo_pair_loss` / `grpo_loss` /
`group_advantages` / `token_logprobs`），缺的是"把它们串成一次训练步"。这里不重写数学，
只做工程：冻结参考模型、清零梯度、反向、裁剪、步进、返回统计。

用户只需提供一个函数：把 `(ids, mask)` 变成 logπ。用 ORV 自己的模型时通常就是

```python
def logprob(ids, mask):
    logits = model(T.tensor(ids))          # [B, T, V]
    return align.seq_logprob(logits, labels)  # 自己按需拼
```

## 边界（如实）

* 只支持**单卡**、同步、单步；不做 rollout 调度、不做分布式、不做采样器。
* `GRPOTrainer` 的 `old_logps` 默认取**第一次调用时的策略输出**（冻结），这是最简做法；
  严格 on-policy 需要每次采样后立即冻结，调用方可用 `set_old_logps` 显式指定。
* 参考模型的 logπ 用 `ref_logprob_fn` 现算（不保留第二份权重）；若想省算力，可传预先算好的
  `ref_logp_*` 常量。
"""
import numpy as np

from . import _as_tensor
from . import align as _align

__all__ = ["DPOTrainer", "GRPOTrainer", "PPOTrainer"]


def _zeros(params):
    for p in params:
        p.grad = None


def _as_np(x):
    """Tensor/ndarray → numpy float32（CPU 上 .grad 可能已是 ndarray）。"""
    if hasattr(x, "to_numpy"):
        return np.asarray(x.to_numpy(), np.float32)
    return np.asarray(x, np.float32)


def _collect_grads(params):
    """把梯度取成 numpy 列表（CPU 上 .grad 可能是 ndarray；CUDA 上是 Tensor）。"""
    out = []
    for p in params:
        g = getattr(p, "grad", None)
        if g is None:
            out.append(None)
        elif hasattr(g, "to_numpy"):
            out.append(np.asarray(g.to_numpy(), np.float32))
        else:
            out.append(np.asarray(g, np.float32))
    return out


class _BaseTrainer(object):
    def __init__(self, params, optimizer=None, max_grad_norm=None):
        self.params = list(params)
        if not self.params:
            raise ValueError("训练器需要至少一个参数" +
                             "。修法：把 model.parameters() 传进来")
        self.optimizer = optimizer
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.steps = 0

    def _finish(self, loss, extra=None):
        _zeros(self.params)
        loss.backward()
        clipped = None
        if self.max_grad_norm is not None:
            clipped = float(_norm_of(self.params))
            from .nn_ops import clip_grad_norm_ as _clip
            _clip(self.params, self.max_grad_norm)
        if self.optimizer is not None:
            self.optimizer.step()
        self.steps += 1
        out = {"loss": float(_align._scalar(loss)), "clipped_norm": clipped, "steps": self.steps}
        if extra:
            out.update(extra)
        return out


def _norm_of(params):
    """全局 L2 范数（用 numpy 汇总，避免依赖具体后端）。"""
    total = 0.0
    for g in _collect_grads(params):
        if g is not None:
            total += float(np.sum(np.square(g.astype(np.float64))))
    return np.float32(np.sqrt(total))


class DPOTrainer(_BaseTrainer):
    """DPO：`logprob_fn(ids, mask) -> 标量 logπ`（可微），`ref_logprob_fn` 同签名取参考。"""

    def __init__(self, params, logprob_fn, ref_logprob_fn=None, beta=0.1,
                 optimizer=None, max_grad_norm=None):
        super(DPOTrainer, self).__init__(params, optimizer, max_grad_norm)
        if float(beta) <= 0:
            raise ValueError("DPO 的 beta 必须为正，收到 %r" % (beta,) +
                             "。修法：常用 0.1 / 0.01")
        self.logprob_fn = logprob_fn
        self.ref_logprob_fn = ref_logprob_fn
        self.beta = float(beta)

    def _ref(self, ids, mask, cached):
        if cached is not None:
            return float(cached)
        if self.ref_logprob_fn is None:
            raise ValueError("没有参考模型的 logπ：请传 ref_logprob_fn，或在 batch 里给 ref_logp_*" +
                             "。修法：ref_logprob_fn 里用冻结的参考模型算一遍")
        return float(_align._scalar(self.ref_logprob_fn(ids, mask)))

    def step(self, chosen_ids, chosen_mask, rejected_ids, rejected_mask,
             ref_logp_w=None, ref_logp_l=None):
        """一次 DPO 步。返回 dict(loss=..., margin=...)。"""
        pw = self.logprob_fn(chosen_ids, chosen_mask)
        pl = self.logprob_fn(rejected_ids, rejected_mask)
        rw = self._ref(chosen_ids, chosen_mask, ref_logp_w)
        rl = self._ref(rejected_ids, rejected_mask, ref_logp_l)
        res = _align.dpo_pair_loss(pw, pl, rw, rl, beta=self.beta)
        loss, stats = res if isinstance(res, tuple) else (res, {})
        margin = (float(_align._scalar(pw)) - rw) - (float(_align._scalar(pl)) - rl)
        extra = {"margin": margin, "ref_logp_w": rw, "ref_logp_l": rl}
        if isinstance(stats, dict):
            extra.update({k: v for k, v in stats.items() if isinstance(v, (int, float))})
        return self._finish(loss, extra)


class GRPOTrainer(_BaseTrainer):
    """GRPO：一组采样 + 奖励 → 优势 → PPO 裁剪损失（可选 KL）。"""

    def __init__(self, params, token_logprob_fn, ref_token_logprob_fn=None,
                 clip_eps=0.2, beta=0.0, kl="k3",
                 optimizer=None, max_grad_norm=None):
        super(GRPOTrainer, self).__init__(params, optimizer, max_grad_norm)
        self.token_logprob_fn = token_logprob_fn
        self.ref_token_logprob_fn = ref_token_logprob_fn
        self.clip_eps = float(clip_eps)
        self.beta = float(beta)
        self.kl = kl
        self._old = None

    def set_old_logps(self, old_logps):
        """显式设置采样时的策略 logπ（常量，冻结）。"""
        self._old = [np.asarray(o, np.float32) for o in old_logps]

    def step(self, ids_list, mask_list, rewards, old_logps=None, ref_logps=None):
        """一组（同一 prompt 的 G 条）GRPO 步。返回 dict(loss=..., adv=...)。"""
        G = len(ids_list)
        if G != len(mask_list):
            raise ValueError("ids_list 与 mask_list 长度不一致（%d vs %d）" % (G, len(mask_list)) +
                             "。修法：每条采样都要有对应的 mask")
        adv = _align.group_advantages(rewards)          # [G]，np.float32
        policy = [self.token_logprob_fn(ids_list[i], mask_list[i]) for i in range(G)]

        if old_logps is not None:
            self.set_old_logps(old_logps)
        if self._old is None:
            # 最简做法：第一次调用时把当前策略输出冻结为 old（后续可用 set_old_logps 覆盖）
            self._old = [np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, np.float32)
                         for x in policy]
        old = [np.asarray(o, np.float32) for o in self._old]        # grpo_loss 内部自己转 Tensor

        if ref_logps is not None:
            ref = [np.asarray(r, np.float32) for r in ref_logps]
        elif self.ref_token_logprob_fn is not None:
            ref = [np.asarray(_as_np(self.ref_token_logprob_fn(ids_list[i], mask_list[i])), np.float32)
                   for i in range(G)]
        else:
            ref = None
        if ref is None:
            if self.beta != 0.0:
                raise ValueError("beta>0 需要参考模型的 logπ：请传 ref_token_logprob_fn 或 ref_logps" +
                                 "。修法：传冻结的参考模型，或把 beta 设为 0")
            ref = old                                   # beta=0 时 KL 项为 0，占位即可

        masks = [np.asarray(m, np.float32) for m in mask_list]
        res = _align.grpo_loss(policy, old, ref, adv, masks,
                               clip_eps=self.clip_eps, beta=self.beta, kl=self.kl)
        loss, gstats = res if isinstance(res, tuple) else (res, {})
        out = self._finish(loss, {"adv": np.asarray(adv, np.float32)})
        if isinstance(gstats, dict):
            out.update({k: v for k, v in gstats.items() if isinstance(v, (int, float))})
        return out


class PPOTrainer(_BaseTrainer):
    """PPO / RLHF 的 actor-critic 训练器（0.9.25）。

    与 `GRPOTrainer` 的区别：GRPO 用**组内**相对优势、不需要 value 网络；
    PPO 需要 critic 给出基线，用 GAE 算优势 —— 这是标准 RLHF 的第二阶段。

    调用方提供两个函数：

    * `token_logprob_fn(ids, mask)` → 逐 token 的 logπ（可微，用于策略损失）
    * `value_fn(ids, mask)` → 逐 token 的 V(s)（可微，用于 critic 损失）

    `step()` 内部顺序：先用**当前**策略算 old logπ 与 V（冻结为常量），再算 GAE，
    最后做一次 PPO 更新。这样同一次调用就是"采样→更新"的最简闭环；
    真实系统里采样由推理引擎完成，届时把 `old_logps` / `old_values` 直接传进来即可。
    """

    def __init__(self, params, token_logprob_fn, value_fn, clip_eps=0.2,
                 vf_coef=0.5, ent_coef=0.01, gamma=0.99, lam=0.95,
                 optimizer=None, max_grad_norm=None, value_params=None):
        super(PPOTrainer, self).__init__(params, optimizer, max_grad_norm)
        from .ppo import ppo_loss
        self._ppo_loss = ppo_loss
        self.token_logprob_fn = token_logprob_fn
        self.value_fn = value_fn
        # critic 的参数并进 self.params：`_finish()` 会统一清零/裁剪/步进
        extra = [p for p in (value_params or []) if p not in self.params]
        if extra:
            self.params = self.params + extra
        self.clip_eps = float(clip_eps)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.gamma = float(gamma)
        self.lam = float(lam)

    def step(self, ids, mask, rewards, dones=None, old_logps=None, old_values=None):
        """一次 PPO 步。返回 dict(loss=..., ratio_mean=..., clip_frac=..., adv=...)。"""
        from .ppo import gae

        # 1) 采样时的策略/价值（冻结为常量）
        if old_logps is None:
            fresh = self.token_logprob_fn(ids, mask)
            old_logps = _as_np(fresh)
        if old_values is None:
            old_values = _as_np(self.value_fn(ids, mask))

        # 2) GAE（常量，不回传）
        adv, ret = gae(rewards, old_values, dones, gamma=self.gamma, lam=self.lam)

        # 3) 用当前策略再算一次 logπ / V / 熵（这次要梯度）
        logps = self.token_logprob_fn(ids, mask)
        vals = self.value_fn(ids, mask)
        m = np.asarray(mask, np.float32)

        loss, stats = self._ppo_loss(logps, old_logps, adv, values=vals,
                                     returns=ret, masks=m, clip_eps=self.clip_eps,
                                     vf_coef=self.vf_coef, ent_coef=0.0)
        out = self._finish(loss, {"adv": np.asarray(adv, np.float32),
                                  "returns": np.asarray(ret, np.float32)})
        if isinstance(stats, dict):
            out.update({k: v for k, v in stats.items() if isinstance(v, (int, float))})
        return out
