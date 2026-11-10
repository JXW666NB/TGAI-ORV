# -*- coding: utf-8 -*-
"""状态复核（一次跑完）：组合算子是否真在 CUDA 上可用 + 关键性能/显存复测。

运行（需独占 GPU）：python status_check.py
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


def rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(1e-9, float(np.abs(b).max())))


def main():
    import torch
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (32, 16)).astype(np.float32)
    mask = (x > 0)
    rep = np.repeat(np.arange(4), 2).astype(np.int64)
    ok = True

    # 组合算子在 CUDA 上是否真的可用（与 CPU 同实现对照）
    cases = [
        ("sqrt", lambda t: (t * t + 1.0).sqrt()),
        ("tanh", lambda t: t.tanh()),
        ("gelu", lambda t: t.gelu()),
        ("masked_fill", lambda t: t.masked_fill(mask, 0.0)),
        ("repeat_interleave", lambda t: t.repeat_interleave(2, dim=0)),
    ]
    for name, fn in cases:
        try:
            T.set_default_device("cpu")
            cpu = np.asarray(fn(T.tensor(x)).to_numpy(), np.float32)
            T.set_default_device("cuda")
            gpu = np.asarray(fn(T.tensor(x)).to_numpy(), np.float32)
            T.set_default_device("cpu")
            r = rel(gpu, cpu)
            log("%-18s CUDA-vs-CPU %.2e %s" % (name, r, "OK" if r <= 1e-6 else "FAIL"))
            ok = ok and r <= 1e-6
        except Exception as exc:
            T.set_default_device("cuda")
            log("%-18s 失败：%s: %s" % (name, type(exc).__name__, str(exc)[:70]))
            ok = False
    T.set_default_device("cuda")

    # repeat_interleave 用 numpy 生成期望值核对（上面是自对照）
    base = np.arange(6, dtype=np.float32).reshape(3, 2)
    want = np.repeat(base, 2, axis=0)
    got = np.asarray(T.tensor(base).repeat_interleave(2, dim=0).to_numpy(), np.float32)
    r = rel(got, want)
    log("repeat_interleave 数值 vs np.repeat %.2e" % r)
    ok = ok and r <= 1e-6

    # GEMM fp32 / fp16 复测
    a_np = rng.normal(0, 0.1, (2048, 2048)).astype(np.float32)
    b_np = rng.normal(0, 0.1, (2048, 6144)).astype(np.float32)

    def bench():
        a, b = T.tensor(a_np), T.tensor(b_np)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(10):
            a @ b
        e1.record(); e1.synchronize()
        return e0.elapsed_time(e1) / 10.0

    t32 = bench()
    with T.amp():
        t16 = bench()
    log("GEMM 2048x2048x6144：fp32 %.2f ms → amp %.2f ms（%.2fx）" % (t32, t16, t32 / max(t16, 1e-9)))

    # 显存：装载 28 层后空闲
    from tgai_go.read_tg import close_tg, read_tg
    from tgai_go.train.v27_train import TGState, V27Trainer
    info = read_tg(MODEL_TG); h = info["header"]; close_tg(info)
    D, NH = int(h["d_model"]), int(h["n_heads"])
    cfg = dict(d_model=D, n_heads=NH, d_ff=int(h["d_ff"]), n_experts=int(h["n_experts"]),
               n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]),
               rope_theta=float(h["rope_theta"]))
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=28)
    tr.load_all(st)
    free = rt.free_device_bytes() / (1 << 30)
    total = rt.total_device_bytes() / (1 << 30) if hasattr(rt, "total_device_bytes") else 31.86
    log("装载 28 层（3.03B）后：空闲 %.2f / %.2f GiB ⇒ 权重约 %.2f GiB"
        % (free, total, total - free))

    log("总结论：%s" % ("PASS" if ok else "FAIL"))
    try:
        st.close(); rt.empty_cache()
    except Exception:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
