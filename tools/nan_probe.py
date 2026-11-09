# -*- coding: utf-8 -*-
"""定位 fused attention 的 NaN：从检查点 numpy 取全宽权重，逐阶段对比。

不依赖库里的 fused_heads（那个已回滚），只用 `blocks.0.attn.*_proj.weight` 原始数组：
  阶段1 投影     per-head x@Wh.T 拼起来  vs  全宽 x@W.T
  阶段2 rope     同左
  阶段3 scores   per-head  vs  全宽切片
  阶段4 输出
每步打印 max|per-head| 与 max|fused| 以及相对差；任何一步出现 nan/inf 就是嫌疑点。

运行（需独占 GPU）：python nan_probe.py
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

MODEL_TG = r"D:\tgai\models\V2.7\TGAI_NB_V2.7_v4.tg"
SEQ = 8            # 小序列，便于打印


def stat(tag, a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    fa = np.isfinite(a).all()
    fb = np.isfinite(b).all()
    if not (fa and fb):
        log("  %-8s 非有限！per-head finite=%s  fused finite=%s" % (tag, fa, fb))
        return False
    rel = float(np.abs(a - b).max() / max(1e-9, float(np.abs(a).max())))
    log("  %-8s max|per-head|=%.4g  max|fused|=%.4g  相对差=%.2e" % (tag, np.abs(a).max(), np.abs(b).max(), rel))
    return rel <= 1e-5


def main():
    import tgai_go.train as T
    from tgai_go.train.v27_train import TGState
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    D, NH = int(h["d_model"]), int(h["n_heads"])
    DH, THETA = D // NH, float(h["rope_theta"])
    T.set_default_device("cuda")
    st = TGState(MODEL_TG)

    qw = np.asarray(st["blocks.0.attn.q_proj.weight"], np.float32)
    kw = np.asarray(st["blocks.0.attn.k_proj.weight"], np.float32)
    vw = np.asarray(st["blocks.0.attn.v_proj.weight"], np.float32)
    ow = np.asarray(st["blocks.0.attn.out_proj.weight"], np.float32)
    log("权重形状 q=%s o=%s（期望都是 [D, D]=[%d, %d]）" % (qw.shape, ow.shape, D, D))

    rng = np.random.default_rng(0)
    x = T.tensor(rng.normal(0, 1, (SEQ, D)).astype(np.float32))
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos_np, sin_np = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    cos, sin = T.tensor(cos_np), T.tensor(sin_np)
    mask = T.tensor(np.triu(np.full((SEQ, SEQ), -1e9, np.float32), k=1))
    inv = 1.0 / np.sqrt(DH)

    # 阶段1：投影
    proj_head = np.concatenate([np.asarray((x @ T.tensor(qw[i * DH:(i + 1) * DH]).T).to_numpy(), np.float32)
                                for i in range(NH)], axis=1)
    q_all_t = T.tensor(qw)
    proj_full = np.asarray((x @ q_all_t.T).to_numpy(), np.float32)
    ok1 = stat("投影", proj_head, proj_full)

    # 阶段2：rope（全宽 + 平铺表）
    rope_head = np.concatenate([np.asarray(T._rope_composed(
        x @ T.tensor(qw[i * DH:(i + 1) * DH]).T, cos, sin).to_numpy(), np.float32) for i in range(NH)], axis=1)
    rope_full = np.asarray(T._rope_composed(x @ q_all_t.T,
                                            T.tensor(np.tile(cos_np, (1, NH))),
                                            T.tensor(np.tile(sin_np, (1, NH)))).to_numpy(), np.float32)
    ok2 = stat("rope", rope_head, rope_full)

    # 阶段3：scores（逐头 vs 切片）
    q_full = T._rope_composed(x @ q_all_t.T, T.tensor(np.tile(cos_np, (1, NH))),
                              T.tensor(np.tile(sin_np, (1, NH))))
    k_full = T._rope_composed(x @ T.tensor(kw).T, T.tensor(np.tile(cos_np, (1, NH))),
                              T.tensor(np.tile(sin_np, (1, NH))))
    s_head = np.concatenate([np.asarray((T._rope_composed(
        x @ T.tensor(qw[i * DH:(i + 1) * DH]).T, cos, sin) @
        T._rope_composed(x @ T.tensor(kw[i * DH:(i + 1) * DH]).T, cos, sin).T).to_numpy(), np.float32
    ).reshape(SEQ, 1, SEQ) for i in range(NH)], axis=1)
    s_full = np.concatenate([np.asarray((q_full[:, i * DH:(i + 1) * DH] @
                                         k_full[:, i * DH:(i + 1) * DH].T).to_numpy(), np.float32
                                        ).reshape(SEQ, 1, SEQ) for i in range(NH)], axis=1)
    ok3 = stat("scores", s_head, s_full)

    log("结论：投影=%s rope=%s scores=%s" % (ok1, ok2, ok3))
    st.close()
    T._cuda().empty_cache()


if __name__ == "__main__":
    main()
