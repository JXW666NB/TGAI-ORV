# -*- coding: utf-8 -*-
"""质量确认（一次装载做完两件事）：

  ① **有限差分梯度检查** —— 补上"数学自洽"唯一的验证缺口。
     解析梯度来自 ORV 的反向（优化器 lr=0，只填梯度不改参数）；
     数值梯度用中心差分 (L(θ+ε) − L(θ−ε)) / 2ε，参数通过替换 `_gpu` 句柄临时改写。

  ② **跨类别 CE 评估** —— 之前只用「思考链」一个文件，这次按 SFT 语料**每个类别取若干条**，
     给出分类别 + 总体 CE（对照 ln(V)=10.40）。

运行（服务必须先停）：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\quality_confirm.py
"""
import glob
import json
import os
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
SFT = "D:\\SFT\u8bed\u6599\u5e93"
GRAD_T = 24          # 梯度检查用的序列长度（短则快，且中心差分更稳）
GRAD_EPS = 1e-3
GRAD_PER_PARAM = 3   # 每个参数抽查几个标量
EVAL_SEQ = 512
EVAL_PER_CAT = 2


def collect_by_category(tk, per_cat=2, seq=512, skip=20):
    """按语料**目录**分类各取若干条，避免只测一个文件带来的偏差。"""
    out = []
    for d in sorted(glob.glob(os.path.join(SFT, "*"))):
        if not os.path.isdir(d):
            continue
        got = 0
        for p in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)):
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    idx = 0
                    for line in f:
                        if idx < skip:
                            idx += 1
                            continue
                        idx += 1
                        try:
                            t = (json.loads(line).get("text") or "")
                        except Exception:
                            continue
                        if len(t) < 120:
                            continue
                        ids = tk.encode(t, add_bos=True)[:seq]
                        if len(ids) >= 64:
                            out.append((os.path.basename(d), ids))
                            got += 1
                        if got >= per_cat:
                            break
            except Exception:
                continue
            if got >= per_cat:
                break
    return out


def main():
    import tgai_go.train as T
    from tgai_go.bpe import BOS_ID, UNK_ID, BpeTokenizer
    from tgai_go.read_tg import close_tg, read_tg
    from tgai_go.train.opt_fp16mv import Fp16MomAdamW
    from tgai_go.train.v27_train import TGState, V27Trainer

    info = read_tg(MODEL)
    h = info["header"]
    tk = BpeTokenizer(info["tokens"], info["merges"], unk_id=UNK_ID, bos_id=BOS_ID)
    close_tg(info)
    samples = collect_by_category(tk, EVAL_PER_CAT, EVAL_SEQ)
    if not samples:
        log("取不到样本，中止")
        raise SystemExit(2)
    cats = sorted({c for c, _ in samples})
    log("样本 %d 条，覆盖 %d 个类别：%s" % (len(samples), len(cats), cats))

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
        log("装载完成 %.0fs，显存 %d MiB" % (time.time() - t0, RUN.vram()))
        emb = st["token_embedding.weight"]
        DH = cfg["d_model"] // cfg["n_heads"]
        theta = cfg["rope_theta"]
        V = int(emb.shape[0])

        def rope(TK):
            j = np.arange(0, DH, 2, dtype=np.float32) / DH
            ang = np.repeat(np.arange(TK, dtype=np.float32)[:, None]
                            * (theta ** (-j))[None, :], 2, axis=1)
            return (T.tensor(np.cos(ang).astype(np.float32)),
                    T.tensor(np.sin(ang).astype(np.float32)))

        def loss_of(ids):
            """只算 loss（不更新参数）。"""
            TK = len(ids) - 1
            cos, sin = rope(TK)
            x0 = np.ascontiguousarray(emb[ids[:-1]], np.float32)
            xs = tr.forward_all(x0, cos, sin)
            loss, _g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
            return float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy") else loss))

        # ---------------- ① 梯度检查 ----------------
        log("=== ① 有限差分梯度检查（T=%d, eps=%g）===" % (GRAD_T, GRAD_EPS))
        ids = np.asarray(samples[0][1][:GRAD_T], np.int64)
        if len(ids) < 8:
            ids = np.random.default_rng(0).integers(0, V, GRAD_T, dtype=np.int64)
        cos, sin = rope(len(ids) - 1)
        x0 = np.ascontiguousarray(emb[ids[:-1]], np.float32)
        xs = tr.forward_all(x0, cos, sin)
        loss, g = tr.loss_and_head_backward(xs[-1], ids[1:], cos, sin)
        # lr=0 ⇒ 只填梯度，不动参数
        zero_opts = [Fp16MomAdamW(tr.params_of(i), lr=0.0) for i in range(cfg["n_layers"])]
        tr.backward_and_step(xs, g, cos, sin, zero_opts)
        zero_head = Fp16MomAdamW(tr.head_params(), lr=0.0)
        log("  解析梯度已生成，loss=%.6f" % float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy") else loss)))

        targets = [
            ("head.embed", tr.head.weight),
            ("final_norm", tr.final_norm.weight),
            ("blocks.0.ln1", tr.layers[0]["ln1"].weight),
            ("blocks.0.moe.router", tr.layers[0]["moe"].router.weight),
            ("blocks.0.exp0.gate", tr.layers[0]["moe"].experts[0].gate.weight),
            ("blocks.27.exp7.down", tr.layers[27]["moe"].experts[7].down.weight),
        ]
        rt = T._cuda()
        worst = 0.0
        for name, p in targets:
            base = np.array(p.to_numpy(), np.float32)
            ana = getattr(p, "grad", None)
            ana_np = None
            if ana is not None:
                try:
                    ana_np = np.asarray(ana.to_numpy() if hasattr(ana, "to_numpy") else ana, np.float32)
                except Exception:
                    ana_np = None
            if ana_np is None or ana_np.shape != base.shape:
                log("  %-22s 跳过（没有梯度）" % name)
                continue
            idxs = np.random.default_rng(3).choice(base.size, GRAD_PER_PARAM, replace=False)
            orig = p._gpu
            rows = []
            try:
                for fi in idxs:
                    plus = base.copy(); plus.reshape(-1)[fi] += GRAD_EPS
                    minus = base.copy(); minus.reshape(-1)[fi] -= GRAD_EPS
                    p._gpu = rt.h2d(plus)
                    lp = loss_of(ids)
                    p._gpu = rt.h2d(minus)
                    lm = loss_of(ids)
                    num = (lp - lm) / (2 * GRAD_EPS)
                    an = float(ana_np.reshape(-1)[fi])
                    rel = abs(num - an) / max(1e-8, abs(num) + abs(an))
                    worst = max(worst, rel)
                    rows.append((fi, an, num, rel))
            finally:
                p._gpu = orig
            for fi, an, num, rel in rows:
                log("  %-22s idx=%-8d 解析=%+.6e 数值=%+.6e 相对误差=%.2e"
                    % (name, fi, an, num, rel))
        log("  梯度检查：最差相对误差 %.2e（判据：<1e-2 视为自洽）" % worst)

        # ---------------- ② 跨类别 CE ----------------
        log("=== ② 跨类别 CE（每类 %d 条，seq<=%d）===" % (EVAL_PER_CAT, EVAL_SEQ))
        per_cat = {}
        for cat, ss in samples:
            try:
                lv = loss_of(np.asarray(ss, np.int64))
            except Exception as e:
                log("  %-16s 评估失败 %s" % (cat, type(e).__name__))
                continue
            per_cat.setdefault(cat, []).append(lv)
            log("  %-16s %4d tok  CE=%.4f" % (cat, len(ss) - 1, lv))
            try:
                rt.empty_cache()
            except Exception:
                pass
        allv = [v for vs in per_cat.values() for v in vs]
        lnv = float(np.log(V))
        log("=== 汇总：%d 条 / %d 类 ===" % (len(allv), len(per_cat)))
        for cat, vs in sorted(per_cat.items()):
            log("  %-16s n=%d 均值 %.4f" % (cat, len(vs), float(np.mean(vs))))
        log("  总体 CE 均值 %.4f（最好 %.4f / 最差 %.4f）  ln(V)=%.2f  ⇒ 好 %.2f 倍"
            % (float(np.mean(allv)), float(np.min(allv)), float(np.max(allv)), lnv, lnv / float(np.mean(allv))))
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
