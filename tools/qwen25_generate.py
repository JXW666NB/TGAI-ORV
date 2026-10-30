# -*- coding: utf-8 -*-
"""Qwen2.5 生成抽检：用我们引擎跑贪心/采样解码，肉眼看输出是否通顺。

为什么需要它：loss 下降只说明"预测下一个 token 更准"，不等于"生成可用"。
微调后必须实际生成几条看看，这是"能用"的必要一环。

实现取舍：**每步完整重算，不做 KV 缓存** —— 慢（0.5B/单卡约 0.3 s/token），
但正确性优先；KV 缓存是后续的优化项。

用法（A10）：
    python tools/qwen25_generate.py --dir /mnt/workspace/qwen25 \
        --load /mnt/workspace/qwen25_sft300.npz \
        --prompt "你是？" --max-new 40 --greedy
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for cand in (os.path.join(REPO, "python"), REPO):
    if os.path.isdir(os.path.join(cand, "tgai_go")) and cand not in sys.path:
        sys.path.insert(0, cand)

PROMPTS = [
    "你是？",
    "介绍一下你自己",
    "1+1等于几？",
]


def load_example():
    path = os.path.join(REPO, "examples", "qwen25_sft.py")
    spec = importlib.util.spec_from_file_location("qwen25_sft", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen25_sft"] = mod
    spec.loader.exec_module(mod)
    return mod


def generate(model, tok, prompt, max_new=32, greedy=True, temperature=0.8, top_k=20):
    """返回 (完整文本, 新生成的 token 数)。"""
    ids = tok.encode(prompt).ids
    if not ids:
        return "", 0
    end_id = None
    try:
        end_id = tok.token_to_id("<|im_end|>")
    except Exception:  # noqa: BLE001
        pass
    out_ids = list(ids)
    new = 0
    for _ in range(max_new):
        xb = np.asarray([out_ids], dtype=np.int64)
        logits = model.forward(xb)
        last = np.asarray(logits.to_numpy(), dtype=np.float32)[0, -1]
        if greedy:
            nxt = int(np.argmax(last))
        else:
            z = last / max(temperature, 1e-6)
            z = z - z.max()
            p = np.exp(z)
            p = p / p.sum()
            if top_k and top_k < p.size:
                keep = np.argpartition(-p, top_k)[:top_k]
                mask = np.zeros_like(p)
                mask[keep] = p[keep]
                p = mask / mask.sum()
            nxt = int(np.random.default_rng().choice(p.size, p=p))
        out_ids.append(nxt)
        new += 1
        if end_id is not None and nxt == end_id:
            break
    text = tok.decode(out_ids)
    return text, new


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/mnt/workspace/qwen25")
    ap.add_argument("--load", default="", help="训练后保存的 npz（缺省用原始权重）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--template", action="store_true",
                    help="按训练时的格式包装提示：<用户>\n{提示}\n<TGAI><回答>\n")
    a = ap.parse_args(argv)

    q = load_example()
    import tgai_go.train as T
    T.set_default_device(a.device)
    cfg = dict(q.CFG)
    w = q.load_safetensors_weights(a.dir, cfg)
    model = q.Qwen25(cfg, w)
    del w
    tie = getattr(model, "tie", None)
    if a.load:
        n = q.load_params(model, a.load)
        print("已加载微调权重 %d 个张量：%s" % (n, a.load))
    else:
        print("使用原始权重（未微调）")
    print("权重绑定: %s" % tie)

    tok = q.load_tokenizer(os.path.join(a.dir, "tokenizer.json"))
    prompts = [a.prompt] if a.prompt else PROMPTS
    for p in prompts:
        q_prompt = ("<用户>\n%s\n<TGAI><回答>\n" % p) if a.template else p
        text, n = generate(model, tok, q_prompt, max_new=a.max_new, greedy=a.greedy)
        print("")
        print("=== 提示：%s" % p)
        print(text.replace("\n", " "))
        print("（新生成 %d token）" % n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
