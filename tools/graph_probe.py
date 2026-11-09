# -*- coding: utf-8 -*-
"""定位捕获期间的非同步点：给 sync / d2h 装探针，打印调用栈后中止。

运行（需独占 GPU）：python graph_probe.py
"""
import sys
import traceback

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

MODEL_TG = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7_v4.tg"
SEQ = 32


class Hit(Exception):
    pass


def install_probes(rt):
    cls = type(rt)
    for name in ("sync", "d2h"):
        orig = getattr(cls, name, None)
        if orig is None:
            continue

        def make(orig_name, orig_fn):
            def probe(self, *a, **kw):
                if getattr(self, "_capturing", False):
                    log("!!! 捕获期间调用了 rt.%s，调用栈：" % orig_name)
                    for line in traceback.format_stack()[-8:]:
                        log("   " + line.strip().replace("\n", " | ")[:200])
                    raise Hit(orig_name)
                return orig_fn(self, *a, **kw)
            return probe

        setattr(cls, name, make(name, orig))
        log("已挂探针: %s" % name)


def main():
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
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=1)
    tr.load_all(st)
    emb = st["token_embedding.weight"]
    rt = T._cuda()

    rng = np.random.default_rng(0)
    ids = rng.integers(0, 30000, SEQ, dtype=np.int64)
    x0 = np.ascontiguousarray(emb[ids], np.float32)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos = T.tensor(np.cos(ang).astype(np.float32))
    sin = T.tensor(np.sin(ang).astype(np.float32))
    in_buf = rt.h2d(x0)
    g_buf = rt.h2d(rng.normal(0, 1, (SEQ, DM)).astype(np.float32))

    def step():
        for p in tr.params_of(0):
            p.grad = None
        x = T.Tensor._from_gpu(in_buf, requires_grad=True)
        y = tr.fwd(x, 0, cos, sin)
        (y * T.Tensor._from_gpu(g_buf)).sum().backward()

    step()                                   # 预热（探针此时不触发）
    rt.sync()
    log("预热完成，开始捕获")

    install_probes(rt)
    ok = rt.graph_arm() and rt.graph_begin()
    log("graph_begin=%s" % ok)
    try:
        step()
        log("捕获期间没有再触发 sync/d2h —— 说明 rc=900 来自别处（可能是 cuBLAS 工作区/h2d）")
        rt.graph_end()
    except Hit as e:
        log("命中：捕获期间调用了 rt.%s（上面是调用栈）" % e)
        rt.graph_abort(str(e))
    st.close()
    rt.empty_cache()


if __name__ == "__main__":
    main()
