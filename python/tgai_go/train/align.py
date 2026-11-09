# -*- coding: utf-8 -*-
"""对齐训练形态（DPO / GRPO）的损失与工具，全部只用 ORV 已有可微算子。

DPO（Rafailov et al. 2023）：
    L = -log σ( β * [ (logπ(y_w|x) − logπ_ref(y_w|x)) − (logπ(y_l|x) − logπ_ref(y_l|x)) ] )
其中 logπ(y|x) 是**回答段**（不含 prompt/padding）的 token log-prob **之和**。

设计取舍（如实说明）：
  后端目前没有 gather 算子，无法一次算出 [B] 形状的"逐样本 logπ 之和"并保持可微。
  所以这里提供的是**单序列标量**版本 + 逐对累加的批版本：
    * `seq_logprob(logits, labels)`   —— 一条序列，标量 Tensor，可微
    * `dpo_pair_loss(pw, pl, rw, rl)` —— 一对偏好数据，标量
    * `dpo_batch_loss(...)`           —— 若干对累加后取平均，全程在计算图上
  代价是 batch 内多几次 kernel（DPO 的批次通常很小，可接受）。
  等后端补上 gather 后可换成向量化实现，接口不必变。
"""
import numpy as np

from . import Tensor, _as_tensor, functional

__all__ = ["logsigmoid", "seq_logprob", "dpo_pair_loss", "dpo_batch_loss"]


def _scalar(t):
    v = t.to_numpy() if hasattr(t, "to_numpy") else np.asarray(t)
    return float(np.asarray(v, np.float32).reshape(-1)[0])


def logsigmoid(x):
    """数值稳定的 log σ(x) = −log(1+exp(−x))。

    用 log σ(x) = −softplus(−x)，softplus(z) = max(z,0) + log(1+exp(−|z|))，避免直接
    `log(1+exp(-x))` 在 |x| 大时溢出。测试里与 torch.nn.functional.logsigmoid 逐元素对照。
    """
    x = _as_tensor(x)
    neg = x * -1.0
    zero = neg * 0.0
    m = neg.maximum(zero)
    absz = neg.maximum(neg * -1.0)
    sp = m + (1.0 + (absz * -1.0).exp()).log()
    return sp * -1.0


def seq_logprob(logits, labels, ignore_index=-100):
    """一条序列回答段的 logπ 之和（标量 Tensor，保持可微）。

    logits: [T, V]；labels: [T]，需要算的位置给真实 token id，其余给 ignore_index。
    实现：`cross_entropy` 对有效位置取**均值**，乘回有效 token 数并取负即为**求和**。
    """
    lab = np.asarray(labels, np.int64)
    valid = int(np.sum(lab != ignore_index))
    if valid == 0:
        raise ValueError("seq_logprob: 该序列没有任何有效 token。修法：检查 loss mask 是否正确")
    ce = functional.cross_entropy(logits, lab, ignore_index=ignore_index)
    return ce * float(valid) * -1.0


def dpo_pair_loss(policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l, beta=0.1):
    """一对偏好数据的 DPO 损失。四个入参都是**标量** logπ 之和（Tensor 或 float）。"""
    beta = float(beta)
    if beta <= 0:
        raise ValueError("DPO 的 beta 必须为正。修法：传 0.1 之类的正数")
    pw = _as_tensor(policy_logp_w)
    pl = _as_tensor(policy_logp_l)
    rw = float(ref_logp_w)
    rl = float(ref_logp_l)
    marg = ((pw - rw) - (pl - rl)) * beta
    loss = logsigmoid(marg) * -1.0
    return loss, marg


def dpo_batch_loss(policy_logits_w, labels_w, policy_logits_l, labels_l,
                   ref_logp_w, ref_logp_l, beta=0.1):
    """若干对偏好数据：逐对算 DPO loss，再取平均（全程在计算图上）。

    ref_logp_w / ref_logp_l 是**参考模型**算好的标量 logπ（用 `seq_logprob` 在
    no_grad 下算的 float 列表）。参考模型必须冻结，这里只接收数值，避免误传梯度。
    """
    n = len(labels_w)
    if not (len(labels_l) == len(policy_logits_w) == len(policy_logits_l) == n):
        raise ValueError("dpo_batch_loss: 四个序列级入参长度不一致（偏好对数量要相同）。"
                         "修法：检查 w/l 两组数据的配对")
    total = None
    margins = []
    for i in range(n):
        pw = seq_logprob(policy_logits_w[i], labels_w[i])
        pl = seq_logprob(policy_logits_l[i], labels_l[i])
        li, marg = dpo_pair_loss(pw, pl, ref_logp_w[i], ref_logp_l[i], beta)
        total = li if total is None else total + li
        margins.append(_scalar(marg))
    return total * (1.0 / float(n)), margins
