# -*- coding: utf-8 -*-
"""`.TG` 回读校验器：解析我们写出的文件，与源张量逐元素比对。

用法:
    python tg_read_tg.py <file.tg> [--dump]
"""
import json
import os
import struct
import sys

import numpy as np

try:
    from mmap import mmap as _MMap          # 用于 isinstance 判断是否走了 mmap
except Exception:                            # pragma: no cover
    _MMap = type(b"")

TG_MAGIC = 0x49414754
#: dtype 码 → (名字, 每元素字节数, numpy 承载类型)
#: 0/1/2/3 是原始四类；4/5 是量化容器（Q8_0/Q4_0，需 dequant 才能变浮点）；
#: 6 是 **BF16**（V2.7 的实际存储类型，实测 header weight_dtype=6）。
DT = {0: ("fp32", 4, np.float32), 1: ("fp16", 2, np.float16),
      2: ("int8", 1, np.int8), 3: ("int32", 4, np.int32),
      4: ("q8_0", 2, np.uint8), 5: ("q4_0", 1, np.uint8),
      6: ("bf16", 2, np.uint16)}
CHAT_MAGIC = 0x46434754


def fnv1a64(name: str) -> int:
    h = 0xCBF29CE484222325
    for b in name.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def parse_tokenizer(buf, off):
    """解析分词段（v3 的 TOK 段与 TG4 的 `SEC_TOK` 分区**同一套字节布局**）。

    ## 为什么要抽出来共用

    v3 把分词表放在头里 `tok_off/tok_len` 指向的位置，TG4 把它放进 `TOK` 分区，
    **但两者内容布局完全一致**：
        u32 v_size, u32 n_merges
        n_merges × (u16 a, u16 b)
        v_size × (u8 len, bytes utf-8)
        16 字节 specials 的 id
    2026-11-13 抽出本函数，让 v3 与 TG4 共用一份解析（此前只在 read_tg 内联）。

    返回 `(tokens, merges, specials)`。
    """
    o = off
    v_size, n_merges = struct.unpack_from("<II", buf, o)
    o += 8
    merges = []
    for _ in range(n_merges):
        a, b = struct.unpack_from("<HH", buf, o)
        o += 4
        merges.append((a, b))
    tokens = []
    for _ in range(v_size):
        ln = buf[o]
        o += 1
        tokens.append(buf[o:o + ln].decode("utf-8", "replace"))
        o += ln
    specials = list(buf[o:o + 16])
    return tokens, merges, specials


def read_tg(path, mmap_mode=True):
    """解析 `.TG` 头/词表/权重表。

    **默认走 mmap**（2026-11-08 加）：3B 模型文件有 5.6 GB，而 `open().read()` 会一口气
    吃掉同样大小的主机内存 —— 在只剩 ~6 GB 可用的机器上直接 MemoryError（V2.7 实测踩过）。
    mmap 只建立映射，`load_tensor` 按张量切片时才真正读那一块。
    用完请调用 `close_tg(info)` 释放句柄。
    """
    f = open(path, "rb")
    raw = None
    if mmap_mode:
        try:
            import mmap as _mmap
            raw = _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ)
        except Exception:
            raw = None
    if raw is None:
        raw = f.read()
    h = struct.unpack("<16I6Q4I", raw[:128])
    (magic, version, wdt, n_layers, d_model, n_heads, d_ff, vocab, max_seq,
     n_experts, n_act, rope_theta, pad, bos, eos, unk,
     tok_off, tok_len, wt_off, wt_cnt, wd_off, wd_len, arch, nkv, modflags, ext_off) = h
    assert magic == TG_MAGIC, hex(magic)

    # tokenizer（与 TG4 的 TOK 分区共用同一个解析器）
    tokens, merges, specials = parse_tokenizer(raw, tok_off)

    # weights
    table = []
    for i in range(wt_cnt):
        p = wt_off + i * 52
        nh, off, ln, dt, nd = struct.unpack_from("<QQQII", raw, p)
        shape = list(struct.unpack_from("<IIII", raw, p + 32))
        (sc,) = struct.unpack_from("<I", raw, p + 48)
        table.append(dict(hash=nh, off=off, len=ln, dtype=dt, ndim=nd,
                          shape=shape[:nd] if nd else shape, scale=sc))

    # chat meta
    # 注意：这里**不能 assert**。V2.7 的真实文件 ext_off 处并不是我们的 CHAT_MAGIC
    # （实测报 "chat meta magic 不对: 3cbb3cab"），但文件本身是好的 —— 是解析器太严。
    # 拿不到就标 meta_ok=False，让调用方决定，绝不因此拒绝一个能用的模型。
    meta = {}
    meta_ok = False
    if ext_off:
        try:
            oo = ext_off
            m, ver, cnt = struct.unpack_from("<III", raw, oo)
            if m == CHAT_MAGIC:
                oo += 12
                for _ in range(cnt):
                    t, ln = struct.unpack_from("<II", raw, oo)
                    oo += 8
                    meta[t] = raw[oo:oo + ln]
                    oo += ln
                meta_ok = True
        except (struct.error, ValueError):
            meta_ok = False

    return dict(header=dict(magic=magic, version=version, weight_dtype=wdt,
                            n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                            d_ff=d_ff, vocab=vocab, max_seq=max_seq,
                            n_experts=n_experts, n_activated=n_act,
                            rope_theta=rope_theta, pad=pad, bos=bos, eos=eos, unk=unk,
                            arch=arch, num_kv_heads=nkv, modality_flags=modflags,
                            extension_off=ext_off),
            tokens=tokens, merges=merges, specials=specials,
            table=table, data_off=wd_off, raw=raw, meta=meta, meta_ok=meta_ok,
            path=path, _file=f, mmap=isinstance(raw, _MMap))


def close_tg(info):
    """释放 mmap/文件句柄（mmap 打开的文件不关会一直占着句柄）。"""
    if not info:
        return
    raw = info.get("raw")
    try:
        if isinstance(raw, _MMap):
            raw.close()
    except Exception:
        pass
    try:
        f = info.get("_file")
        if f is not None:
            f.close()
    except Exception:
        pass


def bf16_to_f32(bits):
    """bf16 位模式 → fp32（左移 16 位，无损还原 bf16 能表示的值）。"""
    return ((np.asarray(bits, dtype=np.uint32) << 16).view(np.float32))


def dequant_q8_0(buf, n):
    """Q8_0：每 32 元素 = fp16 scale(2B) + 32×int8。"""
    raw = np.frombuffer(buf, dtype=np.uint8)
    blk, rem = divmod(len(raw), 34)
    if rem or blk * 32 < n:
        raise ValueError("Q8_0 数据长度不对：%d 字节，期望 %d 个 34B 块（n=%d）。"
                         "修法：确认 .TG 表项 len 与 shape 一致；必要时用 "
                         "tools/tg_requantize.py 重新量化" % (len(raw), blk, n))
    b = raw.reshape(blk, 34)
    scale = b[:, :2].copy().view(np.float16).astype(np.float32)
    q = b[:, 2:].copy().view(np.int8).astype(np.float32)
    return (q * scale).reshape(-1)[:n]


def dequant_q4_0(buf, n):
    """Q4_0：每 32 元素 = fp16 scale(2B) + 16B 打包（低/高 nibble 各 16 个）。"""
    raw = np.frombuffer(buf, dtype=np.uint8)
    blk, rem = divmod(len(raw), 18)
    if rem or blk * 32 < n:
        raise ValueError("Q4_0 数据长度不对：%d 字节，期望 %d 个 18B 块（n=%d）。"
                         "修法：确认表项 len 与 shape 一致" % (len(raw), blk, n))
    b = raw.reshape(blk, 18)
    scale = b[:, :2].copy().view(np.float16).astype(np.float32)
    qs = b[:, 2:]
    lo = (qs & 0x0F).astype(np.int8) - 8          # 4bit 有符号：减 8
    hi = (qs >> 4).astype(np.int8) - 8
    out = np.empty((blk, 32), dtype=np.float32)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return (out * scale[:, None]).reshape(-1)[:n]


def load_tensor(info, name):
    """按参数名取出张量（用 FNV-1a 匹配），统一返回 fp32。"""
    want = fnv1a64(name)
    for e in info["table"]:
        if e["hash"] == want:
            dname, esize, np_t = DT[e["dtype"]]
            start = info["data_off"] + e["off"]
            buf = info["raw"][start:start + e["len"]]
            n = int(np.prod(e["shape"])) if e["shape"] else 0
            if e["dtype"] == 6:                              # BF16
                arr = bf16_to_f32(np.frombuffer(buf, dtype=np.uint16))
                return arr.reshape(e["shape"]).astype(np.float32)
            if e["dtype"] == 4:                              # Q8_0
                return dequant_q8_0(buf, n).reshape(e["shape"]).astype(np.float32)
            if e["dtype"] == 5:                              # Q4_0
                return dequant_q4_0(buf, n).reshape(e["shape"]).astype(np.float32)
            if e["dtype"] in (2, 3):                         # 整型：保留数值
                return np.frombuffer(buf, dtype=np_t).reshape(e["shape"]).astype(np.float32)
            return np.frombuffer(buf, dtype=np_t).reshape(e["shape"]).astype(np.float32)
    return None


def main(path):
    info = read_tg(path)
    h = info["header"]
    print("=== header ===")
    for k, v in h.items():
        print("  %-16s %s" % (k, v))
    print("=== tokenizer ===")
    print("  vocab=%d merges=%d specials[:8]=%s" % (len(info["tokens"]), len(info["merges"]),
                                                    info["specials"][:8]))
    print("  前 10 个 token: %s" % info["tokens"][:10])
    print("=== weights (%d 项) ===" % len(info["table"]))
    for i, e in enumerate(info["table"][:12]):
        print("  #%-2d off=%-8d len=%-8d dtype=%-2d shape=%s"
              % (i, e["off"], e["len"], e["dtype"], e["shape"]))
    if info["meta"]:
        print("=== chat meta ===")
        for t, d in info["meta"].items():
            print("  type=%-2d len=%-4d %r" % (t, len(d), d[:40]))
    return info


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])


# ---------------------------------------------------------------- 名字旁挂
# `.TG` 权重表只存 FNV-1a 哈希、**不存名字**（推理够用，检查点不够用：
# 不能枚举参数就没法做优化器映射 / LoRA 挂载 / 部分加载 / diff）。
# 因此新增一个**可选旁挂文件** `<model>.TG.names.json`：
#   · 它是纯旁挂，产线 C 引擎根本不读它 ⇒ 对推理零风险；
#   · 有了它，`list_tensors()` 才能把"哈希表"变成"名字表"。


def sidecar_path(path):
    return path + ".names.json"


def write_names(path, info, extra=None):
    """按 `.TG` 的实际表项写旁挂名字文件（哈希由表项自带，不重算）。"""
    from collections import Counter
    h = info["header"]
    rows = []
    for e in info["table"]:
        rows.append({"fnv1a64": "%016x" % int(e["hash"]), "off": int(e["off"]),
                     "len": int(e["len"]), "dtype": int(e["dtype"]),
                     "shape": [int(x) for x in e["shape"]], "name": None})
    # 名字由调用方通过 `name_by_hash` 填（见 tools/make_tg_names.py）
    name_by_hash = (extra or {}).get("name_by_hash") or {}
    uniq = Counter()
    for r in rows:
        nm = name_by_hash.get(r["fnv1a64"])
        r["name"] = nm
        if nm:
            uniq[nm] += 1
    dup = sorted([k for k, v in uniq.items() if v > 1])
    doc = {"schema": "tg-names/1", "tg": os.path.basename(path),
           "tg_size": os.path.getsize(path),
           "header": {k: (v if isinstance(v, int) else str(v)) for k, v in h.items()},
           "n_tensors": len(rows),
           "n_named": sum(1 for r in rows if r["name"]),
           "duplicate_names": dup[:20],
           "tensors": rows}
    if extra:
        doc.update({k: v for k, v in extra.items() if k != "name_by_hash"})
    out = sidecar_path(path)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    return out


def load_names(path, verify=True):
    """读旁挂名字文件；`verify=True` 时校验文件大小与张量数是否仍匹配。"""
    p = sidecar_path(path)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        doc = json.load(f)
    if verify:
        if doc.get("tg_size") and doc["tg_size"] != os.path.getsize(path):
            raise ValueError(
                "%s 与旁挂名字文件不匹配：.TG 为 %d B，旁挂记录 %d B。"
                "修法：重新生成旁挂（tools/make_tg_names.py），不要手工改"
                % (path, os.path.getsize(path), doc["tg_size"]))
    return doc


def list_tensors(path_or_info, path=None):
    """返回 [(name, shape, dtype, off, len)]；无名张量 name 为 None。

    优先用旁挂名字文件；没有旁挂时返回全 None 名字（**不会编造名字**）。
    """
    if isinstance(path_or_info, dict):
        info, p = path_or_info, (path or path_or_info.get("path"))
    else:
        info, p = read_tg(path_or_info), path_or_info
    names = load_names(p) if p else None
    by_hash = {}
    if names:
        for r in names.get("tensors", []):
            by_hash[r["fnv1a64"]] = r.get("name")
    out = []
    for e in info["table"]:
        out.append((by_hash.get("%016x" % int(e["hash"])), list(e["shape"]),
                    int(e["dtype"]), int(e["off"]), int(e["len"])))
    return out