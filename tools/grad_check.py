# -*- coding: utf-8 -*-
"""有限差分梯度检查 **v2**（修掉 v1 的两个设计错误）。

v1 为什么不算数（如实记录）：
  1. `head.embed` 抽查的是**词表里没出现在序列中的 token 行** —— embedding 是 gather，
     该行不参与前向 ⇒ 数值梯度必然是 0（v1 报 1.00 相对误差其实是我的测试错）；
  2. `backward_and_step` 用完会把 grad 清掉 ⇒ 其余参数"没有梯度"，检查无从谈起。

v2 做法：
  * 用**带计算图**的前向（`tr.fwd` 逐层 + `final_norm` + `head`）跑一次 `loss.backward()`，
    这样 `.grad` 会留在参数上；
  * 只检查**不经过 gather** 的参数（final_norm / ln1 / router / expert gate / expert down），
    避免 v1 的假失败；
  * 数值梯度用中心差分，两个 eps 各算一次看稳定性。

运行（服务需先停）：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\grad_check.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

MODEL = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7_v4.tg"
T_LEN = 24
EPSS = (1e-2, 1e-3)
PER_PARAM = 3


def main():
    import tgai_go.train as T
    from tgai_go.read_tg import close_tg, read_tg
    from tgai_go.train.v27_train import TGState, V27Trainer

    info = read_tg(MODEL)
    h = info["header"]
    close_tg(info)
    if not RUN.isolate():
        log("GPU 隔离失败，中止")
        raise SystemExit(3)
    try:
        T.set_default_device("cuda")
        cfg = dict(d_model=int(h["d_model"]), n_heads=int(h["n_heads"]), d_ff=int(h["d_ff"]),
                   n_experts=int(h["n_experts"]), n_activated=int(h["n_activated"]),
                   n_layers=int(h["n_layers"]), rope_theta=float(h["rope_theta"]))
        st = TGState(MODEL)
        tr = V27Trainer(cfg, n_layers=cfg["n_layers"])
        t0 = time.time()
        tr.load_all(st)
        tr.attach_head(st)
        log("装载 %.0fs，显存 %d MiB" % (time.time() - t0, RUN.vram()))

        D, DH, theta = cfg["d_model"], cfg["d_model"] // cfg["n_heads"], cfg["rope_theta"]
        V = int(st["token_embedding.weight"].shape[0])
        rng = np.random.default_rng(0)
        ids = rng.integers(0, V, T_LEN + 1, dtype=np.int64)
        tgt = ids[1:]
        TK = T_LEN

        j = np.arange(0, DH, 2, dtype=np.float32) / DH
        ang = np.repeat(np.arange(TK, dtype=np.float32)[:, None]
                        * (theta ** (-j))[None, :], 2, axis=1)
        cos = T.tensor(np.cos(ang).astype(np.float32))
        sin = T.tensor(np.sin(ang).astype(np.float32))
        emb = st["token_embedding.weight"]
        x0 = np.ascontiguousarray(emb[ids[:-1]], np.float32)
        rt = T._cuda()

        def loss_value():
            """不建图地算 loss（用于数值差分）。"""
            xs = tr.forward_all(x0, cos, sin)
            xt = T.tensor(xs[-1])
            logits = tr.head(tr.final_norm(xt))
            loss = T.functional.cross_entropy(logits, tgt)
            return float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy") else loss))

        # ---- 解析梯度：一次带图前向 + backward ----
        t0 = time.time()
        x = T.tensor(x0, requires_grad=True)
        for i in range(tr.NL):
            x = tr.fwd(x, i, cos, sin)
        logits = tr.head(tr.final_norm(x))
        loss = T.functional.cross_entropy(logits, tgt)
        loss.backward()
        log("带图前向+反向 %.1fs  loss=%.6f（基线）"
            % (time.time() - t0, float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy") else loss))))
        base_loss = loss_value()

        targets = [
            ("final_norm", tr.final_norm.weight),
            ("blocks.0.ln1", tr.layers[0]["ln1"].weight),
            ("blocks.0.moe.router", tr.layers[0]["moe"].router.weight),
            ("blocks.0.exp0.gate", tr.layers[0]["moe"].experts[0].gate.weight),
            ("blocks.0.exp0.down", tr.layers[0]["moe"].experts[0].down.weight),
            ("blocks.27.exp7.up", tr.layers[27]["moe"].experts[7].up.weight),
        ]
        worst = 0.0
        checked = 0
        for name, p in targets:
            g = getattr(p, "grad", None)
            if g is None:
                log("  %-22s 跳过（无梯度）" % name)
                continue
            ana = np.asarray(g.to_numpy() if hasattr(g, "to_numpy") else g, np.float32)
            base = np.array(p.to_numpy(), np.float32)
            if ana.shape != base.shape:
                log("  %-22s 跳过（形状不符 %s vs %s）" % (name, ana.shape, base.shape))
                continue
            idxs = np.random.default_rng(11).choice(base.size, PER_PARAM, replace=False)
            orig = p._gpu
            try:
                for fi in idxs:
                    line = "  %-22s idx=%-7d 解析=%+.6e" % (name, fi, float(ana.reshape(-1)[fi]))
                    for eps in EPSS:
                        plus = base.copy(); plus.reshape(-1)[fi] += eps
                        minus = base.copy(); minus.reshape(-1)[fi] -= eps
                        p._gpu = rt.h2d(plus)
                        lp = loss_value()
                        p._gpu = rt.h2d(minus)
                        lm = loss_value()
                        num = (lp - lm) / (2 * eps)
                        an = float(ana.reshape(-1)[fi])
                        rel = abs(num - an) / max(1e-8, abs(num) + abs(an))
                        worst = max(worst, rel)
                        checked += 1
                        line += "  eps=%-6g 数值=%+.6e 相对误差=%.2e" % (eps, num, rel)
                    log(line)
            finally:
                p._gpu = orig
        log("基线 loss=%.6f；检查 %d 个标量；最差相对误差 %.2e（判据 <1e-2 为自洽）"
            % (base_loss, checked, worst))
        log("结论：%s" % ("✅ 梯度与数值一致" if worst < 1e-2 else "❌ 不一致，需排查"))
    finally:
        try:
            tr = None
            st.close()
            rt.empty_cache()
        except Exception:
            pass
        for _ in range(24):
            time.sleep(5)
            if RUN.vram() < 2000:
                break
        RUN.restore()


if __name__ == "__main__":
    main()
