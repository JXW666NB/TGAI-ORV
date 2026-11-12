# -*- coding: utf-8 -*-
"""GPU 续训探针：TG4 检查点 **写 → 读 → 再训一步**，全程在一张卡上跑玩具模型。

## 为什么单独写，而不用 `v27_train_run.py --resume`

`v27_train_run.py` 的 `isolate()` 会**禁用计划任务 + taskkill 产线推理进程（11450/11451）**，
并等显存降到 100 MiB —— 那会打断正在服务的推理，属于需要授权的操作。
本探针只用几十 MB 参数，**不动产线**，专门验证"容器 + 续训机制"这条链：

1. 在 GPU 上建两个参数，跑一步，得到动量；
2. 用真实的 `ckpt.save_checkpoint()` 写检查点（`iter_named_tensors` 用假实现，
   只替掉"从训练器取命名张量"，其余走真代码）；
3. **丢掉一切**，只留磁盘上的目录；
4. 用 `read_tg` + `load_tensor` 按名恢复权重，`ckpt.load_checkpoint()` 恢复优化器；
5. 再跑一步，断言：步数连续（t 恢复）、权重确实被更新、数值有限。

用法：`python tg4_resume_probe.py <带分词表的 .tg 作 tok 源> [--dir D:/tmp/x]`
"""
import argparse
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "python"))

NAMES = ["token_embedding.weight", "blocks.0.attn.q_proj.weight"]
CFG = dict(n_layers=1, d_model=16, n_heads=2, head_dim=8, d_ff=32, vocab=32,
           max_seq=16, n_experts=0, n_activated=0, rope_theta=10000,
           pad=0, bos=1, eos=2, unk=3)
SHAPES = {"token_embedding.weight": (32, 16),
          "blocks.0.attn.q_proj.weight": (16, 16)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_tg", help="提供分词段的 .tg（任意带词表的模型）")
    ap.add_argument("--dir", default="", help="检查点目录（默认在临时目录）")
    a = ap.parse_args()

    import tgai_go.train as T
    from tgai_go import tg4
    from tgai_go.train import ckpt as ck
    from tgai_go.train.opt_fp16mv import Fp16MomAdamW
    from tgai_go.read_tg import read_tg, close_tg, load_tensor, list_tensors

    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)
    host = {n: rng.normal(0, 0.2, SHAPES[n]).astype(np.float32) for n in NAMES}

    # ── 第 1 步：GPU 上建参数 + 优化器，跑一步产生动量 ──────────────
    params = [T.Tensor._from_gpu(rt.h2d(host[n]), requires_grad=True) for n in NAMES]
    opt = Fp16MomAdamW(params, lr=1e-2)

    def one_step():
        opt.zero_grad()
        loss = None
        for p in params:
            term = (p * p).sum()
            loss = term if loss is None else loss + term
        loss.backward()
        opt.step()
        rt.sync()
        return float(np.asarray(loss.to_numpy()).reshape(-1)[0])

    l0 = one_step()
    print("第一步 loss=%.6f  opt.t=%d" % (l0, opt.t))
    before = [np.asarray(p.to_numpy()) for p in params]

    # ── 第 2 步：写真实检查点（只替换"从训练器取命名张量"这一步）──────
    ckpt_dir = a.dir or os.path.join(os.environ.get("TEMP", "/tmp"), "tg4_resume_probe")

    def fake_iter(_tr, _st, want):
        cur = {n: np.asarray(p.to_numpy()) for n, p in zip(NAMES, params)}
        for n in want:
            yield n, cur[n]

    _orig = ck.iter_named_tensors
    ck.iter_named_tensors = fake_iter
    try:
        ck.save_checkpoint(ckpt_dir, None, None, [opt], None, NAMES, a.src_tg,
                           int(opt.t), CFG)
    finally:
        ck.iter_named_tensors = _orig
    wpath = os.path.join(ckpt_dir, ck.CKPT_WEIGHTS)
    with open(wpath, "rb") as fh:
        magic = int.from_bytes(fh.read(4), "little")
    print("检查点: %s (%.1f KB, magic=%#x %s)"
          % (wpath, os.path.getsize(wpath) / 1e3, magic,
             "TG4" if magic == tg4.MAGIC else "**不是 TG4**"))
    if magic != tg4.MAGIC:
        raise SystemExit(2)

    # ── 第 3 步：丢掉内存里的一切，只从磁盘恢复 ─────────────────
    del params, opt
    info = read_tg(wpath)
    try:
        names_got = [r[0] for r in list_tensors(info, wpath)]
        restored = {n: load_tensor(info, n) for n in NAMES}
        n_tok = len(info["tokens"])
    finally:
        close_tg(info)
    print("按名恢复: %d 个名字, 分词表 %d 个" % (len(names_got), n_tok))

    params2 = [T.Tensor._from_gpu(rt.h2d(restored[n]), requires_grad=True) for n in NAMES]
    opt2 = Fp16MomAdamW(params2, lr=1e-2)
    meta = ck.load_checkpoint(ckpt_dir, [opt2], None)
    print("load_checkpoint: step=%s  opt.t=%d" % (meta.get("step"), opt2.t))

    # ── 第 4 步：再跑一步，验证步数连续且权重真的动了 ─────────────
    params = params2
    opt = opt2
    l1 = one_step()
    print("续训一步 loss=%.6f  opt.t=%d" % (l1, opt.t))
    after = [np.asarray(p.to_numpy()) for p in params2]
    moved = [float(np.abs(a2 - b).max()) for a2, b in zip(after, before)]

    ok = True
    if opt2.t < 2:
        print("  ✗ 步数没有连续（t=%d）" % opt2.t); ok = False
    if not np.isfinite([l0, l1]).all():
        print("  ✗ loss 非有限"); ok = False
    if max(moved) < 1e-6:
        print("  ✗ 权重没动（max|Δ|=%.2e）" % max(moved)); ok = False
    if n_tok == 0:
        print("  ✗ 检查点里没有分词表"); ok = False
    print("判读:", "TG4 检查点可续训 PASS" if ok else "**FAIL**")
    if not a.dir:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
