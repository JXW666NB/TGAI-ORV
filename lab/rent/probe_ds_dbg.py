# -*- coding: utf-8 -*-
"""定位：为什么 CSA 的 dS / dp 捕获到全 0。

做两件事：
  1. 隔离测试 `_cap`（clone + 自定义 _backward）在 4D + softmax + matmul 上是否正确；
  2. 在真实训练 step 2 上，给 CSA layer0 全局分支的
     `out`（attn 输出）、`p`、`scores` 各插一个捕获节点，打印各自 dL/d* 的 absmax，
     同时打印前向 p/v/scores 的 absmax。
"""
import os
import sys
import traceback

WORK = r"D:\orv_bench"
os.chdir(WORK)
sys.path.insert(0, WORK)

import numpy as np                                              # noqa: E402

import tgai_go.train as T                                       # noqa: E402
from tgai_go.train import optim as _optim                       # noqa: E402
from tgai_go.train import v3_pretrain as TR                     # noqa: E402
from tgai_go.train import v3_csa as C                           # noqa: E402

RT = T._cuda()
print("[dbg] device ready", flush=True)

rng = np.random.default_rng(0)


def _dev(a):
    return T.Tensor._from_gpu(RT.h2d(np.asarray(a, np.float32)))


def _h(t):
    return np.asarray(t.to_numpy(), np.float64)


def trial(shape, label):
    a = rng.standard_normal(shape).astype(np.float32) * 0.3
    vsh = shape[:-2] + (shape[-1], 8)
    v = rng.standard_normal(vsh).astype(np.float32) * 0.5
    x = _dev(a)
    x.requires_grad = True
    s = x * 0.5
    cap = {}
    buf = RT.clone(s._gpu)
    xc = T.Tensor._from_gpu(buf, requires_grad=True, _parents=(s,))
    print("   cap shape=%r vs s shape=%r" % (tuple(xc.shape), tuple(s.shape)), flush=True)

    def bw(g, grads):
        cap["ds"] = _h(T.Tensor._from_gpu(g, requires_grad=False))
        T._acc(grads, s, g)
    xc._backward = bw
    p = T.functional.softmax(xc, axis=-1)
    y = p @ _dev(v)
    # dL/dy = e_0 ⇒ dp[a,c] = v[c,0]（对 a 广播）
    loss = y[:, :, :, 0].sum() * 1.0
    loss.backward()
    got = cap.get("ds")
    pf = np.exp(a * 0.5)
    pf = pf / pf.sum(-1, keepdims=True)
    dp = np.broadcast_to(v[..., 0][..., None, :], pf.shape).astype(np.float64)
    ds = pf * (dp - (dp * pf).sum(-1, keepdims=True))
    print("[%s] cap.ds absmax=%.6e  analytic absmax=%.6e  relFro=%.3e"
          % (label, np.abs(got).max() if got is not None else -1,
             np.abs(ds).max(),
             np.linalg.norm((got if got is not None else 0) - ds) /
             max(np.linalg.norm(ds), 1e-30)), flush=True)


trial((1, 2, 8, 16), "4d")
print("[dbg] isolated _cap test done", flush=True)

# ---------------------------------------------------------------- 真实训练
STEP = [0]
CUR = {"step": 0, "csa": 0}
TARGET = {2}
INFO = []


def _hmax(buf):
    a = np.asarray(T.Tensor._from_gpu(buf, requires_grad=False).to_numpy(), np.float32)
    return float(np.abs(a).max()) if a.size else -1.0


def _cap(x, key):
    buf = RT.clone(x._gpu)
    out = T.Tensor._from_gpu(buf, requires_grad=True, _parents=(x,))

    def bw(g, grads):
        try:
            INFO.append((CUR["step"], key, _hmax(g)))
        except Exception:                                        # noqa: BLE001
            traceback.print_exc()
        T._acc(grads, x, g)
    out._backward = bw
    return out


_CSA_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10]
_orig_attn = C._attn


def _attn_probe(q, k, v, mask, row_mask, scale):
    idx = CUR["csa"]
    CUR["csa"] = idx + 1
    li = idx // 2
    layer = _CSA_LAYERS[li] if li < len(_CSA_LAYERS) else -1
    local = (idx % 2 == 1)
    want = (CUR["step"] in TARGET) and (layer == 0) and (not local)
    scores = (q @ k.swap_last2()) * float(scale)
    scores = scores + mask
    if want:
        print("   [fw] csa l0: q=%r k=%r v=%r mask=%r row=%r scores_absmax=%.4e"
              % (tuple(q.shape), tuple(k.shape), tuple(v.shape),
                 tuple(mask.shape) if mask is not None else None,
                 tuple(row_mask.shape) if row_mask is not None else None,
                 _hmax(scores._gpu)), flush=True)
        print("   [fw]   q_absmax=%.3e k_absmax=%.3e v_absmax=%.3e scale=%r"
              % (_hmax(q._gpu), _hmax(k._gpu), _hmax(v._gpu), scale), flush=True)
        scores = _cap(scores, "ds")
    p = T.functional.softmax(scores, axis=-1)
    if want:
        print("   [fw]   p_absmax=%.6e" % _hmax(p._gpu), flush=True)
        p = _cap(p, "dp")
    out = p @ v
    if row_mask is not None:
        out = out * row_mask
    if want:
        print("   [fw]   out_absmax=%.6e" % _hmax(out._gpu), flush=True)
        out = _cap(out, "dout")
    return out


C._attn = _attn_probe

_orig_sched_step = _optim.lr_scheduler.CosineLR.step


def _sched_step(self):
    STEP[0] += 1
    return _orig_sched_step(self)


_optim.lr_scheduler.CosineLR.step = _sched_step

_orig_fwd = TR.forward_loss


def _fwd(P, cfg, kinds, cos, sin, ids, labels, seg, **kw):
    CUR["step"] = STEP[0] + 1
    CUR["csa"] = 0
    loss, meta = _orig_fwd(P, cfg, kinds, cos, sin, ids, labels, seg, **kw)
    if CUR["step"] in TARGET:
        print("[fwd] step %d loss %.6f"
              % (CUR["step"], float(np.asarray(loss.to_numpy()).reshape(-1)[0])), flush=True)
    return loss, meta


TR.forward_loss = _fwd

argv = ["--data", os.path.join(WORK, "data2048_v96000"), "--vocab", "96000",
        "--seq", "2048", "--d", "1024", "--layers", "12", "--E", "4",
        "--n_q", "16", "--d_ff", "3328", "--top_k", "2", "--csa", "4",
        "--opt", "int8gpu", "--amp", "gemm", "--batch", "1",
        "--steps", "3", "--log-every", "99", "--no-final-save"]
TR.main(argv)
print("\n[dbg] captures:")
for r in INFO:
    print("   step=%s %-6s absmax=%.6e" % r)
