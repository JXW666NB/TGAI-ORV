# -*- coding: utf-8 -*-
"""验证 GGUF → .TG 转换的正确性：按 FNV-1a 哈希把 .TG 张量取回，与 GGUF 源逐张量对比。

三个关键点（都踩过）：
  1. `.TG` 的权重表**不存名字**，只存 FNV-1a 哈希 ⇒ 必须用转换器同一套命名映射；
     所以这里**直接 import `tools/gguf_to_tg.py` 的 `map_weight_name`**，绝不各写一套。
  2. 转换器会**重排 Q/K 的 RoPE 维度**（rotate-half → 引擎交错式）⇒
     q/k 及其 bias 与 GGUF **本来就不该逐位相同**，只校验「值集合相同」（排序后比对），
     以证明是重排而不是写坏。
  3. `output.weight` 被有意跳过（与 token_embedding 权重绑定），不算缺失。

用法：
    python tools/verify_gguf_tg.py --gguf X.gguf --tg X.TG
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

from tgai_go.read_tg import DT, fnv1a64, read_tg  # noqa: E402

ROPE_PERMUTED = ("attn_q", "attn_k")     # 这两个会被重排（含 .bias）


def load_converter():
    path = os.path.join(REPO, "tools", "gguf_to_tg.py")
    spec = importlib.util.spec_from_file_location("gguf_to_tg", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gguf_to_tg"] = mod
    spec.loader.exec_module(mod)
    return mod


def read_one(info, entry):
    _n, _s, np_t = DT[entry["dtype"]]
    start = info["data_off"] + entry["off"]
    buf = info["raw"][start:start + entry["len"]]
    return np.frombuffer(buf, dtype=np_t).reshape(entry["shape"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tg", required=True)
    ap.add_argument("--arch", default="", help="默认从 GGUF 元数据读 general.architecture")
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args(argv)

    conv = load_converter()
    import gguf

    rd = gguf.GGUFReader(a.gguf)
    meta = {}
    for k in rd.fields:
        try:
            v = rd.fields[k].contents()
            if isinstance(v, (bytes, bytearray)):
                v = bytes(v).decode("utf-8", "replace")
            meta[k] = v
        except Exception:  # noqa: BLE001
            pass
    arch = a.arch or str(meta.get("general.architecture", "llama"))
    if arch not in conv.ARCH_CONFIG:
        print("不支持的 arch：%s（支持 %s）" % (arch, list(conv.ARCH_CONFIG)))
        return 1
    cfg = conv.ARCH_CONFIG[arch]
    print("arch=%s（arch_type=%s）" % (arch, cfg["arch_type"]))

    info = read_tg(a.tg)
    hdr, table = info["header"], info["table"]
    by_hash = {e["hash"]: e for e in table}
    print("== .TG ==  version=%s dtype=%s vocab=%s layers=%s 张量表=%d 词表=%d merges=%d"
          % (hdr["version"], hdr["weight_dtype"], hdr["vocab"], hdr["n_layers"],
             len(table), len(info["tokens"]), len(info["merges"])))

    ok = bad = skipped = 0
    shown = 0
    for gname in rd.tensors_mapped if hasattr(rd, "tensors_mapped") else [t.name for t in rd.tensors]:
        t = next(t for t in rd.tensors if t.name == gname)
        ref = np.asarray(t.data)
        mapped = conv.map_weight_name(gname, cfg)
        if mapped is None:
            skipped += 1
            continue
        e = by_hash.get(fnv1a64(mapped))
        if e is None:
            print("   [取不到] %-34s -> %s" % (gname, mapped))
            bad += 1
            continue
        got = read_one(info, e)
        if got.size != ref.size:
            print("   [大小不符] %-34s TG=%s GGUF=%s" % (mapped, got.shape, ref.shape))
            bad += 1
            continue
        g = got.reshape(-1).astype(np.float64)
        r = ref.reshape(-1).astype(np.float64)
        if any(k in gname for k in ROPE_PERMUTED):
            # 只校验「同一组值，只是顺序变了」
            gs, rs = np.sort(g), np.sort(r)
            d = float(np.abs(gs - rs).max())
            rel = d / (float(np.abs(r).max()) or 1.0)
            tag = "重排"
        else:
            d = float(np.abs(g - r).max())
            rel = d / (float(np.abs(r).max()) or 1.0)
            tag = "逐位"
        if rel <= 2e-3:
            ok += 1
        else:
            bad += 1
            if bad <= 5:
                print("   [数值不符] %-34s(%s) max|Δ|=%.3e rel=%.2e" % (mapped, tag, d, rel))
        if shown < a.show:
            print("   %-40s %-4s shape=%-14s max|Δ|=%.3e rel=%.2e"
                  % (mapped, tag, str(tuple(e["shape"][:e["ndim"]])), d, rel))
            shown += 1

    print("")
    print("结果：一致 %d / 不符 %d / 有意跳过 %d（output.weight 权重绑定）" % (ok, bad, skipped))
    if bad:
        print("结论：FAIL")
        return 1
    print("结论：PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
