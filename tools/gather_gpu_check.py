# -*- coding: utf-8 -*-
"""gather/index_select 的 **CUDA 路径**核对（新内核 + atomicAdd 反向）。

CPU 路径已由 tests/test_indexing.py 覆盖；这里专门验证刚加的 4 个内核：
gather_rows / scatter_add_rows / gather_flat / scatter_add_flat。

运行（需独占 GPU）：
    python gather_gpu_check.py
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

F = None


def rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))


def main():
    global F
    import torch
    import torch.nn.functional as Fn
    F = Fn
    import tgai_go.train as T

    T.set_default_device("cuda")
    log("device=%s" % T.get_default_device() if hasattr(T, "get_default_device") else "cuda")

    rng = np.random.default_rng(0)
    ok = True

    # 1) index_select（dim=0，含重复索引 → 反向 atomicAdd 累加）
    x = rng.normal(0, 1, (16, 8)).astype(np.float32)
    idx = np.array([3, 1, 1, 1, 15, 0], np.int64)
    w = rng.normal(0, 1, (6, 8)).astype(np.float32)

    xt = T.tensor(x.copy(), requires_grad=True)
    y = T.index_select(xt, 0, idx)
    (y * T.tensor(w)).sum().backward()
    ours = np.asarray(y.to_numpy(), np.float32)
    og = np.asarray(xt.grad.to_numpy(), np.float32)

    xr = torch.from_numpy(x.copy()).cuda().requires_grad_(True)
    yr = torch.index_select(xr, 0, torch.from_numpy(idx).cuda())
    (yr * torch.from_numpy(w).cuda()).sum().backward()
    r_f = rel(ours, yr.detach().cpu().numpy())
    r_g = rel(og, xr.grad.cpu().numpy())
    log("index_select 前向 相对=%.2e  反向 相对=%.2e" % (r_f, r_g))
    ok = ok and r_f <= 1e-6 and r_g <= 1e-6

    # 2) index_select（dim=1，走 outer=1 的行 gather 变形）
    x2 = rng.normal(0, 1, (3, 9)).astype(np.float32)
    idx2 = np.array([8, 0, 4], np.int64)
    o2 = np.asarray(T.index_select(T.tensor(x2), 1, idx2).to_numpy(), np.float32)
    r2 = rel(o2, torch.index_select(torch.from_numpy(x2), 1, torch.from_numpy(idx2)).numpy())
    log("index_select dim=1 前向 相对=%.2e" % r2)
    ok = ok and r2 <= 1e-6

    # 3) gather_flat（含重复下标 → 反向累加）
    x3 = rng.normal(0, 1, (4, 6)).astype(np.float32)
    fi = np.array([0, 7, 7, 23, 5], np.int64)
    w3 = rng.normal(0, 1, (5,)).astype(np.float32)
    xt3 = T.tensor(x3.copy(), requires_grad=True)
    y3 = T.gather_flat(xt3, fi)
    (y3 * T.tensor(w3)).sum().backward()

    flat = torch.from_numpy(x3.copy()).cuda().reshape(-1).requires_grad_(True)
    y3r = flat[torch.from_numpy(fi).cuda()]
    (y3r * torch.from_numpy(w3).cuda()).sum().backward()
    r3f = rel(np.asarray(y3.to_numpy(), np.float32), y3r.detach().cpu().numpy())
    r3g = rel(np.asarray(xt3.grad.to_numpy(), np.float32), flat.grad.reshape(4, 6).cpu().numpy())
    log("gather_flat 前向 相对=%.2e  反向 相对=%.2e" % (r3f, r3g))
    ok = ok and r3f <= 1e-6 and r3g <= 1e-6

    # 4) token_logprobs（B3：走 gather/scatter，无 [T,V] one-hot）前向 + 梯度
    Tlen, V = 6, 11
    lg = rng.normal(0, 1, (Tlen, V)).astype(np.float32)
    lab = rng.integers(0, V, Tlen).astype(np.int64)
    lab[:2] = -100
    w4 = rng.normal(0, 1, (Tlen,)).astype(np.float32)
    lgt = T.tensor(lg.copy(), requires_grad=True)
    tl = T.token_logprobs(lgt, lab)
    (tl * T.tensor(w4)).sum().backward()
    ours4 = np.asarray(tl.to_numpy(), np.float32)
    grad4 = np.asarray(lgt.grad.to_numpy(), np.float32)

    lpt = torch.log_softmax(torch.from_numpy(lg.copy()).cuda(), dim=-1)
    it = torch.from_numpy(np.where(lab == -100, 0, lab)).cuda().long()
    pk = lpt.gather(1, it[:, None]).squeeze(1)
    m4 = torch.from_numpy((lab != -100).astype(np.float32)).cuda()
    ref4 = (pk * m4 * torch.from_numpy(w4).cuda())
    lg_leaf = torch.from_numpy(lg.copy()).cuda().requires_grad_(True)
    lpt2 = torch.log_softmax(lg_leaf, dim=-1)
    pk2 = lpt2.gather(1, it[:, None]).squeeze(1)
    (pk2 * m4 * torch.from_numpy(w4).cuda()).sum().backward()
    r4f = rel(ours4, (pk * m4).detach().cpu().numpy())
    r4g = rel(grad4, lg_leaf.grad.cpu().numpy())
    log("token_logprobs 前向 相对=%.2e  反向 相对=%.2e" % (r4f, r4g))
    ok = ok and r4f <= 1e-6 and r4g <= 1e-6

    # 5) log_softmax（曾经因为缺 _warp_row_grid 在 GPU 上直接崩）前向 + 反向
    lg5 = rng.normal(0, 1, (7, 13)).astype(np.float32)
    w5 = rng.normal(0, 1, (7, 13)).astype(np.float32)
    t5 = T.tensor(lg5.copy(), requires_grad=True)
    y5 = T.nn.functional.log_softmax(t5, axis=-1)
    (y5 * T.tensor(w5)).sum().backward()
    ours5 = np.asarray(y5.to_numpy(), np.float32)
    g5 = np.asarray(t5.grad.to_numpy(), np.float32)

    t5r = torch.from_numpy(lg5.copy()).cuda().requires_grad_(True)
    y5r = torch.log_softmax(t5r, dim=-1)
    (y5r * torch.from_numpy(w5).cuda()).sum().backward()
    r5f = rel(ours5, y5r.detach().cpu().numpy())
    r5g = rel(g5, t5r.grad.cpu().numpy())
    log("log_softmax 前向 相对=%.2e  反向 相对=%.2e" % (r5f, r5g))
    ok = ok and r5f <= 1e-6 and r5g <= 1e-6

    log("CUDA gather 结论：%s" % ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
