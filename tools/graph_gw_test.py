# -*- coding: utf-8 -*-
"""关键假设验证：图重放时是否**读取**上游梯度缓冲（决定每步能否换梯度）。

做法：同一层捕获一张前反向图，缓冲 gw_buf 固定；先写入 gw1 重放→记录梯度；
再**只改缓冲内容**为 gw2，重放→记录梯度；另跑一次 eager(gw2) 对比。
若 replay(gw2) == eager(gw2) 且 != replay(gw1)，说明梯度是运行时读取的 ⇒ 每步可换。

运行（需独占 GPU）：python graph_gw_test.py
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
SEQ = 64


def main():
    import torch
    import tgai_go.train as T
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
    tr = V27Trainer(cfg, n_layers=1)
    tr.load_all(st)
    L = tr.layers[0]

    rng = np.random.default_rng(0)
    x_np = rng.normal(0, 1, (SEQ, D)).astype(np.float32)
    gw1 = rng.normal(0, 1, (SEQ, D)).astype(np.float32)
    gw2 = rng.normal(0, 5, (SEQ, D)).astype(np.float32)
    x_buf = rt.h2d(x_np)
    gw_buf = rt.h2d(gw1)                       # 固定缓冲，捕获时用它
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos, sin = T.tensor(np.cos(ang).astype(np.float32)), T.tensor(np.sin(ang).astype(np.float32))
    mask = tr._causal_mask(SEQ)
    inv = 1.0 / np.sqrt(DH)

    def h2d_into(buf, arr):
        """把 host 数组写进**已有**设备缓冲（运行时只提供 h2d，它会新建缓冲）。"""
        import ctypes
        a = np.ascontiguousarray(arr, np.float32)
        rt._activate()
        rt._chk(rt.cu.cuMemcpyHtoD(ctypes.c_uint64(int(buf.ptr)),
                                   a.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_size_t(a.nbytes)), "cuMemcpyHtoD")

    def block():
        x = T.Tensor._from_gpu(x_buf, requires_grad=True)
        hh = L["ln1"](x)
        acc = None
        for t in range(NH):
            q = T._rope_composed(L["q"][t](hh), cos, sin)
            k = T._rope_composed(L["k"][t](hh), cos, sin)
            v = L["v"][t](hh)
            pr = T.functional.softmax((q @ k.T) * inv + mask, axis=-1)
            part = L["o"][t](pr @ v)
            acc = part if acc is None else acc + part
        (acc * T.Tensor._from_gpu(gw_buf)).sum().backward()

    def zero():
        for k in ("q", "k", "v", "o"):
            for m in L[k]:
                shp = tuple(int(x) for x in m.weight.shape)
                m.weight.grad = T.tensor(np.zeros(shp, np.float32))

    def gsum():
        return sum(float(np.abs(np.asarray(m.weight.grad.to_numpy(), np.float32)).sum())
                   for k in ("q", "k", "v", "o") for m in L[k])

    # eager(gw2)
    h2d_into(gw_buf, gw2)
    zero(); block(); rt.sync(); e2 = gsum()
    log("eager(gw2) 梯度合计 = %.6f" % e2)

    # 捕获（用 gw1）
    zero()
    rt.reserve_capture_arena(512 * 1024 ** 2)
    if not (rt.graph_arm() and rt.graph_begin()):
        log("捕获不可用"); return 2
    block()
    if not rt.graph_end():
        log("graph_end 失败"); return 2

    # 重放一次（缓冲仍是 gw2 的内容，因为上面已写入）
    rt.graph_launch_slot(0); rt.sync()
    r2 = gsum()
    log("replay(gw2) 梯度合计 = %.6f  （与 eager 相对差 %.2e）"
        % (r2, abs(r2 - e2) / max(1e-9, abs(e2))))

    # 再把缓冲写成 gw1（值不同），重放
    h2d_into(gw_buf, gw1)
    rt.graph_launch_slot(0); rt.sync()
    r1 = gsum()
    zero(); block(); rt.sync(); e1 = gsum()
    log("replay(gw1) 梯度合计 = %.6f  （eager(gw1) = %.6f，相对差 %.2e）"
        % (r1, e1, abs(r1 - e1) / max(1e-9, abs(e1))))
    changed = abs(r1 - r2) / max(1e-9, abs(r1))
    log("换缓冲后结果是否变化：%.2e（>1e-3 说明梯度是运行时读取的 ⇒ 每步可换）" % changed)
    try:
        rt.graph_release_all(); st.close(); rt.empty_cache()
    except Exception as e:
        log("清理异常（忽略）：%s" % type(e).__name__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
