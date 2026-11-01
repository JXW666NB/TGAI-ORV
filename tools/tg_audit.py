# -*- coding: utf-8 -*-
"""`.TG` 加载体检：同一份模型，用 `.TG` 和 safetensors 各算一遍，比 CE。

为什么需要它：`.TG` 是我们自己的格式，但转换器版本/布局约定可能不一致 ——
实测踩过：某份 `.TG` 的线性层是 `[in, out]`，加载后**能训但训的是垃圾**
（base probe 10.2 vs 1.81，gnorm 4237 vs 28），而**没有任何报错**。

这条工具把"有没有读对"变成一个**可执行、可比对**的动作：

    python tools/tg_audit.py --tg model.TG --safetensors /path/Qwen2.5-0.5B \
        --tokenizer /path/tokenizer.json [--device cuda]

判定（写死，避免事后解释）：
  * 两者的 prompt CE 相差 ≤ 0.5  → PASS（同一份权重）
  * 相差 > 0.5                  → FAIL，并提示最可能的原因（布局/置换/量化）
  * 任一侧加载失败               → FAIL，打印各自的错误
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

PROMPTS = ["The capital of France is", "中国的首都是北京。"]


def _ce(model, tok, text):
    ids = np.asarray(tok.encode(text).ids, np.int64).reshape(1, -1)
    lg = np.asarray(model.forward(ids).to_numpy(), np.float64)[0][:-1]
    tgt = ids[0, 1:]
    lg = lg - lg.max(-1, keepdims=True)
    lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    return float(-lp[np.arange(len(tgt)), tgt].mean())


def compare(ce_a, ce_b, tol=0.5):
    """纯函数：给两组 CE，返回 (ok, 说明)。便于单测。"""
    if ce_a is None or ce_b is None:
        return False, "有一侧加载/前向失败，无法比对"
    diff = abs(float(ce_a) - float(ce_b))
    if diff <= tol:
        return True, "两份权重等价（CE 差 %.3f ≤ %.1f）" % (diff, tol)
    return False, ("两份权重不等价（CE %.3f vs %.3f，差 %.3f > %.1f）。"
                   "最可能原因：① `.TG` 线性层是 [in,out] 布局（需要转置）；"
                   "② q/k 的 RoPE 通道置换方向不对；③ 权重是量化/半精度但有缩放没还原"
                   % (ce_a, ce_b, diff, tol))


def _build(loader, path, tok_path, device):
    import tgai_go.train as T
    from tgai_go.train import checkpoints as C
    T.set_default_device(device)
    sys.path.insert(0, os.path.join(REPO, "examples"))
    import qwen25_sft as Q
    ck = loader(path)
    cfg = Q.cfg_from_checkpoint(ck)
    model = Q.Qwen25(cfg, dict(ck.weights))
    tok = Q.load_tokenizer(tok_path)
    del ck
    return model, tok, cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tg", required=True)
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=float, default=0.5)
    a = ap.parse_args(argv)

    from tgai_go.train import checkpoints as C

    res = {}
    for tag, loader, path in (("tg", C.load_tg, a.tg),
                              ("safetensors", C.load_safetensors_dir, a.safetensors)):
        try:
            model, tok, cfg = _build(loader, path, a.tokenizer, a.device)
            ces = [_ce(model, tok, p) for p in PROMPTS]
            res[tag] = (sum(ces) / len(ces), ces, cfg)
            print("%-12s CE=%s  (L=%d D=%d NH=%d KV=%d DH=%d)"
                  % (tag, " ".join("%.3f" % c for c in ces),
                     cfg["L"], cfg["D"], cfg["NH"], cfg["KV"], cfg["DH"]))
        except Exception as e:                       # noqa: BLE001
            res[tag] = (None, [], None)
            print("%-12s 失败：%s: %s" % (tag, type(e).__name__, str(e)[:160]))

    ok, why = compare(res["tg"][0], res["safetensors"][0], a.tol)
    print("结论：%s —— %s" % ("PASS" if ok else "FAIL", why))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
