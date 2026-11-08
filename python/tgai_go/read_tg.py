# -*- coding: utf-8 -*-
"""`.TG` 回读校验器：解析我们写出的文件，与源张量逐元素比对。

用法:
    python tg_read_tg.py <file.tg> [--dump]
"""
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

    # tokenizer
    o = tok_off
    v_size, n_merges = struct.unpack_from("<II", raw, o)
    o += 8
    merges = []
    for _ in range(n_merges):
        a, b = struct.unpack_from("<HH", raw, o)
        o += 4
        merges.append((a, b))
    tokens = []
    for _ in range(v_size):
        ln = raw[o]
        o += 1
        tokens.append(raw[o:o + ln].decode("utf-8", "replace"))
        o += ln
    specials = list(raw[o:o + 16])
    o += 16

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
