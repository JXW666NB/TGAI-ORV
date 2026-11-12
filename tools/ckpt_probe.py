# -*- coding: utf-8 -*-
"""诊断：梯度检查点到底有没有释放层内激活。

## 判据（决定性）

固定 B/S/D/dh，只改层数 L：
* **真的在释放** ⇒ 峰值活张量与 L 基本无关（只多出残差流那点线性增长）
* **没在释放** ⇒ 峰值随 L 线性增长，斜率≈单层激活大小

同时对比 checkpoint 开 / 关 两条曲线。若两条重合，说明检查点等于没开。

用法：python ckpt_probe.py
"""
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_pkg_root(start):
    cur = start
    for _ in range(6):
        if os.path.isdir(os.path.join(cur, "python", "tgai_go")):
            return os.path.join(cur, "python")
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    raise SystemExit("找不到 python/tgai_go")


sys.path.insert(0, _find_pkg_root(HERE))
OUT = os.path.join(HERE, "ckptprobe.txt")

D, NH, FF, S = 1280, 10, 5120, 2048
DH = D // NH
VOCAB = 32000


def one(L, use_ckpt, bsz=1):
    import tgai_go.train as T
    from tgai_go.train import functional as Fn

    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)

    def dev(n, m):
        return T.Tensor._from_gpu(rt.h2d(rng.normal(0, 0.02, (n, m)).astype(np.float32)))

    qkv = [dev(3 * D, D) for _ in range(L)]
    o = [dev(D, D) for _ in range(L)]
    gate = [dev(FF, D) for _ in range(L)]
    up = [dev(FF, D) for _ in range(L)]
    down = [dev(D, FF) for _ in range(L)]
    for t in qkv + o + gate + up + down:
        t.requires_grad = True

    ids = rng.integers(0, VOCAB, (bsz, S)).astype(np.int64)
    emb = T.Tensor._from_gpu(rt.h2d(
        rng.normal(0, 0.02, (bsz * S, D)).astype(np.float32))).reshape(bsz, S, D)
    emb.requires_grad = True

    def block(x, i):
        B, Sq, _ = x.shape
        z = x.reshape(B * Sq, D) @ qkv[i].T
        q, k, v = z.reshape(B, Sq, 3, NH, DH).permute(2, 0, 3, 1, 4)
        q = q.reshape(B * NH, Sq, DH)
        k = k.reshape(B * NH, Sq, DH)
        v = v.reshape(B * NH, Sq, DH)
        y = Fn.scaled_dot_product_attention(q, k, v, is_causal=True, lowmem=True)
        y = y.reshape(B, NH, Sq, DH).permute(0, 2, 1, 3).reshape(B * Sq, D)
        x = x + (y @ o[i].T).reshape(B, Sq, D)
        f = x.reshape(B * Sq, D)
        h = Fn.silu(f @ gate[i].T) * (f @ up[i].T)
        h = h @ down[i].T
        return x + h.reshape(B, Sq, D)

    rt.pool_stats()          # 初始化
    base = rt.pool_stats()["live_bytes"]
    x = emb
    for i in range(L):
        if use_ckpt:
            x = T.checkpoint(lambda h, _i=i: block(h, _i), x)
        else:
            x = block(x, i)
    peak_fwd = rt.pool_stats()["peak_live_bytes"]
    loss = (x.reshape(-1, D) @ x.reshape(-1, D).T).reshape(-1)[0]
    loss.backward()
    rt.sync()
    peak_all = rt.pool_stats()["peak_live_bytes"]
    return (peak_fwd - base) / (1 << 30), (peak_all - base) / (1 << 30)


def main():
    rep = ["配置：D=%d NH=%d dh=%d FF=%d S=%d B=1" % (D, NH, DH, FF, S), ""]
    rep.append("| 层数 L | 检查点 | 前向峰值(增量 GiB) | 前向+反向峰值(增量 GiB) |")
    rep.append("| --- | --- | --- | --- |")
    for L in (2, 4, 8):
        for ck in (False, True):
            try:
                f, a = one(L, ck)
                rep.append("| %d | %s | %.2f | %.2f |" % (L, "开" if ck else "关", f, a))
            except Exception as ex:      # noqa: BLE001
                rep.append("| %d | %s | 失败 %s | - |" % (L, "开" if ck else "关",
                                                          type(ex).__name__))
    rep.append("")
    rep.append("判读：**检查点若有效，前向峰值应几乎不随 L 增长**；")
    rep.append("若「开」与「关」两行数值接近，说明检查点等于没开。")
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
