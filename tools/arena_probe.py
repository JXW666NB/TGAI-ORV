# -*- coding: utf-8 -*-
"""探测 T=512 下**每张图真正需要多少 arena**（二分），以及 28 张能否放下。

背景：拆分显示整步 1412 ms 里注意力反向占 638 ms（最大头），全 28 层走图估算是 862 ms。
卡点是每图 arena 459 MiB × 28 = 13.1 GiB，模型加载后只剩 20.6 GiB。

如果实际需要远小于 459 MiB（arena_need 量的是高水位，可能含可避免的拷贝），就能全捕获。

运行（需独占 GPU）：python arena_probe.py
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
SEQ = 512
ARENAS_MIB = [192, 256, 320, 384, 460]


def main():
    import torch
    import tgai_go.train as T
    from tgai_go.train.layer_graph import LayerGraphBackward
    from tgai_go.train.v27_train import TGState, V27Trainer
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    D, NH = int(h["d_model"]), int(h["n_heads"])
    DH, THETA = D // NH, float(h["rope_theta"])
    cfg = dict(d_model=D, n_heads=NH, d_ff=int(h["d_ff"]), n_experts=int(h["n_experts"]),
               n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]), rope_theta=THETA)
    T.set_default_device("cuda")
    rt = T._cuda()
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=2)
    tr.load_all(st)

    rng = np.random.default_rng(0)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos, sin = T.tensor(np.cos(ang).astype(np.float32)), T.tensor(np.sin(ang).astype(np.float32))
    xs = [rng.normal(0, 1, (SEQ, D)).astype(np.float32) for _ in range(2)]
    gw = rng.normal(0, 1, (SEQ, D)).astype(np.float32)

    # 逐算子参照（用于校验图重放是否真的对）
    def eager():
        for i in range(2):
            for k in ("q", "k", "v", "o"):
                for m in tr.layers[i][k]:
                    m.weight.grad = T.tensor(np.zeros(tuple(int(x) for x in m.weight.shape), np.float32))
        out = []
        for i in range(2):
            xi = T.tensor(xs[i], requires_grad=True)
            ya = tr.fwd_attn(xi, i, cos, sin)
            (ya * T.tensor(gw)).sum().backward()
            out.append(np.asarray(xi.grad.to_numpy(), np.float32))
        return out

    ref = eager()
    rt.sync()
    log("逐算子参照完成；装载后空闲 %.2f GiB" % (rt.free_device_bytes() / (1 << 30)))

    for mib in ARENAS_MIB:
        ok = True
        try:
            rt.empty_cache()
            gb = LayerGraphBackward(tr, cos, sin, seq=SEQ, arena_bytes=mib * 1024 ** 2)
            gb.capture(xs, max_graphs=2)
            got = [gb.replay_layer(i, gw) for i in range(2)]
            rt.sync()
            for i in range(2):
                d = float(np.abs(got[i] - ref[i]).max() / max(1e-9, float(np.abs(ref[i]).max())))
                ok = ok and d <= 1e-6
            gb.close()
            log("arena=%4d MiB -> 捕获+重放成功，dL/dx 最大相对差 %s ⇒ %s"
                % (mib, "%.2e" % d, "PASS" if ok else "FAIL"))
        except Exception as exc:
            log("arena=%4d MiB -> 失败：%s: %s" % (mib, type(exc).__name__, str(exc)[:90]))
            try:
                gb.close()
            except Exception:
                pass
        try:
            rt.empty_cache()
        except Exception:
            pass

    log("若更小的 arena 也能 PASS，就可把 28 张图压进显存（当前用 480 MiB/张）。")
    try:
        st.close(); rt.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
