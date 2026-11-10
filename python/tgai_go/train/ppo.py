# -*- coding: utf-8 -*-
"""PPO / RLHF 的损失件（0.9.25）：GAE 与 PPO 裁剪目标。

## 与 GRPO 的分工

* `grpo_loss`：**组内**相对优势，不需要 value 网络（省一份模型）。
* 本模块 `ppo_loss`：需要 **critic（value 网络）** 给出基线，用 GAE 算优势 —— 这是标准 RLHF 的第二阶段。

二者都只做"损失"这一层，训练循环在 `rl_trainer.PPOTrainer`。

## 公式（与主流实现一致）

GAE（广义优势估计）：
```
δ_t = r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t)
A_t = δ_t + γ·λ·(1−done_t)·A_{t+1}
returns_t = A_t + V(s_t)
```

PPO 裁剪目标（逐 token 平均）：
```
r_t = exp(logπ_new − logπ_old)
L_pg  = −min(r_t·A_t, clip(r_t,1−ε,1+ε)·A_t)
L_vf  = (V_new − returns)²
L_ent = −entropy
L = L_pg + vf_coef·L_vf + ent_coef·L_ent
```

## 数值稳定

* 优势做标准化（`normalize_adv=True`，与多数实现一致）；
* `L_vf` 可选裁剪（`vf_clip`），避免价值函数一次更新过猛；
* 优势与 returns 都是**常量**（不回传），只有 logπ_new 与 V_new 需要梯度。
"""
import numpy as np

from . import _as_tensor

__all__ = ["gae", "ppo_loss", "discounted_returns"]


def gae(rewards, values, dones=None, gamma=0.99, lam=0.95):
    """广义优势估计。全 numpy，返回 `(advantages, returns)`。

    参数
      rewards : [T] 每步奖励
      values  : [T] 每步价值 V(s_t)
      dones   : [T] 0/1，1 表示该步是终止（不 bootstrap）。None 视为全 0
      gamma   : 折扣因子
      lam     : GAE 的 λ

    注意：`values` 是**当前** critic 的输出，调用方应在更新 critic 前取。
    """
    r = np.asarray(rewards, np.float64).reshape(-1)
    v = np.asarray(values, np.float64).reshape(-1)
    if r.size != v.size:
        raise ValueError("gae: rewards(%d) 与 values(%d) 长度不一致" % (r.size, v.size) +
                         "。修法：每个时间步都要有对应的价值估计")
    if r.size == 0:
        raise ValueError("gae: 序列为空" + "。修法：至少给一个时间步")
    d = np.zeros_like(r) if dones is None else np.asarray(dones, np.float64).reshape(-1)
    if d.size != r.size:
        raise ValueError("gae: dones 长度应与 rewards 一致" + "。修法：检查 mask")

    adv = np.zeros_like(r)
    last = 0.0
    for t in range(r.size - 1, -1, -1):
        next_v = v[t + 1] if t + 1 < r.size else 0.0
        nonterminal = 1.0 - d[t]
        delta = r[t] + float(gamma) * next_v * nonterminal - v[t]
        last = delta + float(gamma) * float(lam) * nonterminal * last
        adv[t] = last
    returns = adv + v
    return adv.astype(np.float32), returns.astype(np.float32)


def discounted_returns(rewards, gamma=0.99):
    """只用奖励的折扣回报（没有 critic 时的最简基线）。"""
    r = np.asarray(rewards, np.float64).reshape(-1)
    out = np.zeros_like(r)
    acc = 0.0
    for t in range(r.size - 1, -1, -1):
        acc = r[t] + float(gamma) * acc
        out[t] = acc
    return out.astype(np.float32)


def ppo_loss(policy_logps, old_logps, advantages, values=None, old_values=None,
             returns=None, entropy=None, masks=None,
             clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, vf_clip=None,
             normalize_adv=True):
    """PPO 损失。返回 `(loss, stats)`。

    `policy_logps` / `values` / `entropy` 是**可微**张量（`[T]`），其余是常量。
    给了 `values` 就必须给 `returns`（critic 的回归目标）。
    """
    e = float(clip_eps)
    if e <= 0:
        raise ValueError("ppo 的 clip_eps 必须为正，收到 %r" % (clip_eps,) +
                         "。修法：常用 0.2")
    p = _as_tensor(policy_logps)
    old = _as_tensor(np.asarray(old_logps, np.float32)).to(getattr(p, "device", "cpu"))
    adv_np = np.asarray(advantages, np.float32).reshape(-1)
    if normalize_adv and adv_np.size > 1:
        sd = float(adv_np.std())
        adv_np = (adv_np - float(adv_np.mean())) / (sd + 1e-8)
    adv = _as_tensor(adv_np).to(getattr(p, "device", "cpu"))

    m = None
    if masks is not None:
        m = _as_tensor(np.asarray(masks, np.float32)).to(getattr(p, "device", "cpu"))

    ratio = (p - old).exp()
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - e, 1.0 + e) * adv
    # min(s1, s2) —— **不要**用 maximum 组合实现：
    # 当 ratio == 1 时 s1 == s2 是平局，`maximum` 的梯度会整份给第二个参数
    # ⇒ 代理损失的梯度符号翻转，策略被推向反方向（实测：优势为正的样本 logπ 反而下降）。
    # 用**显式掩码**挑较小者：平局时两者相等，取哪边导数都正确。
    s1n = np.asarray(s1.to_numpy() if hasattr(s1, "to_numpy") else s1, np.float32)
    s2n = np.asarray(s2.to_numpy() if hasattr(s2, "to_numpy") else s2, np.float32)
    take1 = (s1n <= s2n).astype(np.float32)
    dev = getattr(p, "device", "cpu")
    m1 = _as_tensor(take1).to(dev)
    mn = s1 * m1 + s2 * (1.0 - m1)
    pg = mn * -1.0
    if m is not None:
        pg = pg * m
    # token 数从 **numpy 入参** 算：`np.asarray(Tensor)` 在 ORV 上不可靠，Tensor.sum() 也不能直接 float()
    _n_old = float(np.asarray(old_logps, np.float32).size)
    ntok = float(np.asarray(masks, np.float32).sum()) if masks is not None else _n_old
    loss = pg.sum() * (1.0 / max(ntok, 1e-9))

    stats = {"clip_frac": 0.0, "ratio_mean": 0.0}

    # 价值损失
    if values is not None:
        if returns is None:
            raise ValueError("ppo_loss: 给了 values 就要给 returns（critic 的回归目标）" +
                             "。修法：用 gae() 的返回值")
        v = _as_tensor(values)
        ret = _as_tensor(np.asarray(returns, np.float32)).to(getattr(v, "device", "cpu"))
        v_err = v - ret
        if vf_clip is not None and old_values is not None:
            v_old = _as_tensor(np.asarray(old_values, np.float32)).to(getattr(v, "device", "cpu"))
            v_clipped = v_old + (v - v_old).clamp(-float(vf_clip), float(vf_clip))
            v_err = (v - ret).maximum(v_clipped - ret)   # 取较大的误差（保守）
        vf = (v_err * v_err)
        if m is not None:
            vf = vf * m
        loss = loss + float(vf_coef) * (vf.sum() * (1.0 / max(ntok, 1e-9)))

    # 熵奖励（entropy 越大越好 ⇒ 损失里减）
    if entropy is not None and float(ent_coef) != 0.0:
        ent = _as_tensor(entropy)
        if m is not None:
            ent = ent * m
        loss = loss - float(ent_coef) * (ent.sum() * (1.0 / max(ntok, 1e-9)))

    # 统计
    rv = np.asarray(ratio.to_numpy() if hasattr(ratio, "to_numpy") else ratio, np.float32)
    stats["ratio_mean"] = float(rv.mean())
    hit = np.abs(rv - 1.0) > e
    if m is not None:
        mv = np.asarray(m.to_numpy() if hasattr(m, "to_numpy") else m, np.float32)
        stats["clip_frac"] = float((hit * (mv > 0)).sum() / max(mv.sum(), 1e-9))
    else:
        stats["clip_frac"] = float(hit.mean())
    stats["adv_mean"] = float(adv_np.mean())
    stats["tokens"] = ntok
    return loss, stats
