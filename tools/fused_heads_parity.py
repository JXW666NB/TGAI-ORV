# -*- coding: utf-8 -*-
"""默认（per-head）vs `fused_heads=True`（全宽投影）的对照：等价性 + 发射数 + 耗时。

前提：先应用补丁 `python tools/apply_fused_heads.py`，否则 `fused_heads` 会被忽略，
两侧结果相同（脚本会检测到并用退出码 2 提示）。

运行（需独占 GPU）：python fused_heads_parity.py
"""
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

MODEL_TG = r"D:\tgai\models\V2.7\TGAI_NB_V2.7_v4.tg"
SEQ = 128
REP = 5


def main():
    import torch
    import tgai_go.train as T
    from tgai_go.train.v27_train import TGState, V27Trainer
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    DM, NH = int(h["d_model"]), int(h["n_heads"])
    DH, THETA = DM // NH, float(h["rope_theta"])
    cfg = dict(d_model=DM, n_heads=NH, d_ff=int(h["d_ff"]), n_experts=int(h["n_experts"]),
               n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]), rope_theta=THETA)
    T.set_default_device("cuda")

    rng = np.random.default_rng(0)
    ids = rng.integers(0, 30000, SEQ, dtype=np.int64)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos = T.tensor(np.cos(ang).astype(np.float32))
    sin = T.tensor(np.sin(ang).astype(np.float32))
    # 保留 numpy 版本：CUDA 张量没有可用的主机镜像，不能用 .data 取数组
    cos_np = np.cos(ang).astype(np.float32)
    sin_np = np.sin(ang).astype(np.float32)

    counts = Counter()
    rt = T._cuda()
    orig = type(rt).launch

    def counted(self, name, *a, **kw):
        counts[name] += 1
        return orig(self, name, *a, **kw)

    type(rt).launch = counted

    def make(fused):
        tr = V27Trainer(cfg, n_layers=1)
        tr.fused_heads = bool(fused)          # 必须在 load_all 之前
        st = TGState(MODEL_TG)
        tr.load_all(st)
        return tr, st

    def attn_block(tr, x_np):
        """只跑 attention 部分（跳过 MoE），便于隔离投影改动。"""
        x = T.tensor(x_np, requires_grad=True)
        L = tr.layers[0]
        hh = L["ln1"](x)
        inv = 1.0 / np.sqrt(tr.DH)
        if getattr(tr, "fused_heads", False):
            # 平铺表每个 trainer 只建一次：循环内反复分配会和池/free 交互，可能读到已释放的块
            if not hasattr(tr, "_cos_t"):
                tr._cos_t = T.tensor(np.tile(cos_np, (1, tr.NH)))
                tr._sin_t = T.tensor(np.tile(sin_np, (1, tr.NH)))
            cos_t, sin_t = tr._cos_t, tr._sin_t
            q_all = T._rope_composed(L["q"][0](hh), cos_t, sin_t)
            k_all = T._rope_composed(L["k"][0](hh), cos_t, sin_t)
            v_all = L["v"][0](hh)
            acc = None
            for i in range(tr.NH):
                sl = slice(i * tr.DH, (i + 1) * tr.DH)
                pr = T.functional.softmax(
                    (q_all[:, sl] @ k_all[:, sl].T) * inv + tr._causal_mask(q_all.shape[0]), axis=-1)
                part = (pr @ v_all[:, sl]) @ L["o"][0].weight[:, sl].T
                acc = part if acc is None else acc + part
        else:
            acc = None
            for i in range(tr.NH):
                q = T._rope_composed(L["q"][i](hh), cos, sin)
                k = T._rope_composed(L["k"][i](hh), cos, sin)
                v = L["v"][i](hh)
                pr = T.functional.softmax((q @ k.T) * inv + tr._causal_mask(SEQ), axis=-1)
                part = L["o"][i](pr @ v)
                acc = part if acc is None else acc + part
        return acc, x

    x0 = np.ascontiguousarray(np.asarray(np.random.default_rng(1).normal(0, 1, (SEQ, DM)),
                                         np.float32))

    tr_a, st_a = make(False)
    counts.clear(); ya, xa = attn_block(tr_a, x0); rt.sync(); c_a = sum(counts.values())
    counts.clear(); yb, xb = attn_block(tr_a, x0); rt.sync(); c_b = sum(counts.values())
    if c_a != c_b:
        log("!! 同一路径两次发射数不同（%d vs %d），计时不可比" % (c_a, c_b))
    out_default = np.asarray(ya.data, np.float32).copy()
    st_a.close()

    tr_f, st_f = make(True)
    if int(tr_f.layers[0]["q"][0].weight.shape[0]) != DM:
        log("!! 补丁未应用：fused_heads 被忽略（q[0] 形状 %s）。先跑 tools/apply_fused_heads.py"
            % (tuple(int(s) for s in tr_f.layers[0]["q"][0].weight.shape),))
        st_f.close()
        return 2
    counts.clear(); yc, xc = attn_block(tr_f, x0); rt.sync(); c_f = sum(counts.values())
    out_fused = np.asarray(yc.data, np.float32).copy()
    st_f.close()

    rel = float(np.abs(out_default - out_fused).max() / max(1e-9, float(np.abs(out_default).max())))
    log("输出相对差 = %.2e（判据 1e-6）" % rel)
    log("发射次数：默认 %d，fused %d（少 %d，%.1f%%）"
        % (c_a, c_f, c_a - c_f, 100.0 * (c_a - c_f) / max(c_a, 1)))

    def timeit(tr):
        attn_block(tr, x0); torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(REP):
            attn_block(tr, x0)
        torch.cuda.synchronize()
        return (time.time() - t0) / REP

    tr_a2, st_a2 = make(False)
    tr_f2, st_f2 = make(True)
    t_def, t_fus = timeit(tr_a2), timeit(tr_f2)
    log("耗时：默认 %.2f ms，fused %.2f ms（省 %.2f ms）"
        % (t_def * 1000, t_fus * 1000, (t_def - t_fus) * 1000))
    st_a2.close(); st_f2.close()
    rt.empty_cache()
    log("结论：%s" % ("PASS（等价）" if rel <= 1e-6 else "FAIL（不等价，必须回滚）"))
    return 0 if rel <= 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main())
