# -*- coding: utf-8 -*-
"""`.TG` **V4** 读写（自描述分区，推理 + 训练统一）。

设计见 `docs/TG_V4_FORMAT.md`。要点：
* **头 128B 定长**，字段只读；新需求走 **section table**，不改头、不改读者；
* **NAM 名字表**：名字 + shape + dtype + WGT 内偏移 ⇒ 可枚举（v3 只有哈希，做不到）；
* **WGT 每张量 64B 对齐** ⇒ 单张量 mmap 切片即用；
* 训练态独立分区 `TRAIN` / `OPT`，可只放独立文件，避免把推理文件撑大；
* 读端遇到**不认识的 section 直接跳过**（这是能长期演进的关键）。

dtype 码沿用 v3：0=fp32 1=fp16 2=int8 6=bf16。
"""
import hashlib
import json
import mmap
import os
import struct

import numpy as np

MAGIC = 0x00344754          # b"TG4\0"
VERSION = 4
HEADER_BYTES = 128
DT_FP32, DT_FP16, DT_INT8, DT_BF16 = 0, 1, 2, 6
_ESZ = {DT_FP32: 4, DT_FP16: 2, DT_INT8: 1, DT_BF16: 2}
#: section 类型（u32）
SEC_TOK, SEC_NAM, SEC_WGT, SEC_ARC, SEC_TRAIN, SEC_OPT, SEC_HASH = range(1, 8)
SECNAME = {SEC_TOK: "TOK", SEC_NAM: "NAM", SEC_WGT: "WGT", SEC_ARC: "ARC",
           SEC_TRAIN: "TRAIN", SEC_OPT: "OPT", SEC_HASH: "HASH"}
ALIGN = 64
_HDR = struct.Struct("<16I4I6Q4I")     # 64 + 16 + 48 + 16 = 144? -> 见 _pack_header 校验
_SEC = struct.Struct("<IIQQII")        # type/flags/off/len/align/crc32 = 32B
_NAM = struct.Struct("<I H B B Q 4I")  # name_off/len/ndim/dtype/data_off/shape[4] = 32B


def _pack_header(h):
    b = bytearray(HEADER_BYTES)
    f = struct.pack_into
    f("<16I", b, 0, MAGIC, VERSION, HEADER_BYTES, h.get("flags", 0),
      h["n_layers"], h["d_model"], h["n_heads"], h["head_dim"],
      h["d_ff"], h["vocab"], h["max_seq"], h["n_kv_heads"],
      h["n_experts"], h["n_activated"], h["arch"], h.get("weight_dtype", DT_BF16))
    f("<4f", b, 64, float(h.get("rope_theta", 10000.0)), 0.0, 0.0, 0.0)
    f("<4I", b, 80, h.get("pad_id", 0), h.get("bos_id", 0),
      h.get("eos_id", 0), h.get("unk_id", 0))
    f("<Q", b, 96, h.get("tok_off", 0))
    f("<Q", b, 104, h.get("tok_len", 0))
    f("<Q", b, 112, h.get("sect_off", 0))
    f("<Q", b, 120, h.get("sect_cnt", 0))
    return bytes(b)


def _unpack_header(b):
    (magic, ver, hb, flags, L, D, NH, HD, FF, V, MS, NKV, NE, NA, arch, wdt) = \
        struct.unpack_from("<16I", b, 0)
    if magic != MAGIC:
        raise ValueError("不是 TG4 文件（magic=%x）。修法：v3 文件请用 read_tg()；"
                         "TG4 头前 4 字节应为 'TG4\\0'" % magic)
    if ver != VERSION:
        raise ValueError("TG4 版本 %d 不支持（本读者只认 %d）。修法：升级 tg4.py"
                         % (ver, VERSION))
    rope = struct.unpack_from("<f", b, 64)[0]
    pad, bos, eos, unk = struct.unpack_from("<4I", b, 80)
    tok_off, tok_len = struct.unpack_from("<2Q", b, 96)
    sect_off, sect_cnt = struct.unpack_from("<2Q", b, 112)
    return dict(magic=magic, version=ver, header_bytes=hb, flags=flags,
                n_layers=L, d_model=D, n_heads=NH, head_dim=HD, d_ff=FF, vocab=V,
                max_seq=MS, n_kv_heads=NKV, n_experts=NE, n_activated=NA, arch=arch,
                weight_dtype=wdt, rope_theta=rope, pad_id=pad, bos_id=bos,
                eos_id=eos, unk_id=unk, tok_off=tok_off, tok_len=tok_len,
                sect_off=sect_off, sect_cnt=sect_cnt)


def _pad_to(buf, align=ALIGN):
    r = len(buf) % align
    if r:
        buf += b"\x00" * (align - r)


def _enc(arr, dtype):
    a = np.ascontiguousarray(arr, dtype=np.float32)
    if dtype == DT_BF16:
        # 就近舍入（round-half-to-even）：产线 .TG 用的是 RU，截断会差 0.5 ulp
        # 实测：fp32 源的 112 个注意力张量，截断 vs 产线 差 1.95e-3；改 RU 后 0 差异
        u = a.view(np.uint32).astype(np.uint64)
        u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
        return u.astype(np.uint16).tobytes()
    if dtype == DT_FP16:
        return a.astype(np.float16).tobytes()
    if dtype == DT_INT8:
        return np.clip(np.rint(a), -127, 127).astype(np.int8).tobytes()
    return a.tobytes()


def _dec(buf, dtype, shape):
    if dtype == DT_BF16:
        v = (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif dtype == DT_FP16:
        v = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
    elif dtype == DT_INT8:
        v = np.frombuffer(buf, dtype=np.int8).astype(np.float32)
    else:
        v = np.frombuffer(buf, dtype=np.float32)
    return v.reshape(shape)


def write_tg4(path, tensors, cfg, tok=None, arc=None, train=None, per_tensor_hash=False):
    """写 TG4。`tensors`: {name: ndarray}；`cfg`: 架构字段（见 _pack_header）。

    布局（确定性）：
        128B 头 → section table（每条 32B，自然对齐）→ 各分区（每段起点 64B 对齐）
    张量在 WGT 内的偏移由 NAM 记录，且每个张量起点也做 64B 对齐。
    """
    names = sorted(tensors)                      # 稳定顺序，便于逐位比对
    dt_file = cfg.get("weight_dtype", DT_BF16)
    wgt = bytearray()
    ent_meta = []
    for n in names:
        a = np.ascontiguousarray(tensors[n], dtype=np.float32)
        _pad_to(wgt)                             # 每个张量 64B 对齐
        res = {"off": len(wgt), "dtype": dt_file, "shape": tuple(a.shape)}
        wgt += _enc(a, dt_file)
        ent_meta.append((n, res))

    # NAM 用 JSON：982 个名字约 100 KB，换来"不会错位"（二进制池的越界解码已踩过）
    nam = json.dumps(
        [{"name": n, "off": r["off"], "dtype": r["dtype"], "shape": list(r["shape"])}
         for n, r in ent_meta], ensure_ascii=False).encode("utf-8")

    payloads = [(SEC_TOK, tok or b""), (SEC_NAM, nam), (SEC_WGT, bytes(wgt))]
    if arc is not None:
        payloads.append((SEC_ARC, json.dumps(arc, ensure_ascii=False).encode("utf-8")))
    if train is not None:
        payloads.append((SEC_TRAIN, json.dumps(train, ensure_ascii=False).encode("utf-8")))
    if per_tensor_hash:
        hb = bytearray()
        for n, res in ent_meta:
            sz = int(np.prod(res["shape"])) * _ESZ[res["dtype"]]
            hb += hashlib.sha256(bytes(wgt[res["off"]:res["off"] + sz])).digest()
        payloads.append((SEC_HASH, bytes(hb)))

    out = bytearray(HEADER_BYTES + _SEC.size * len(payloads))
    _pad_to(out)
    entries = []
    for typ, data in payloads:
        _pad_to(out)
        entries.append((typ, 0, len(out), len(data), ALIGN, 0))
        out += data

    h = dict(cfg)
    h["tok_off"], h["tok_len"] = 0, 0
    h["sect_off"], h["sect_cnt"] = HEADER_BYTES, len(entries)
    out[0:HEADER_BYTES] = _pack_header(h)
    for i, e in enumerate(entries):
        _SEC.pack_into(out, HEADER_BYTES + i * _SEC.size, *e)
    with open(path, "wb") as f:
        f.write(out)
    return path


class TG4(object):
    """TG4 读端（mmap；`get()` 按名字取单个张量，不整文件入内存）。"""

    def __init__(self, path):
        self.path = path
        self._f = open(path, "rb")
        self._m = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self.header = _unpack_header(self._m[:HEADER_BYTES])
        cnt = self.header["sect_cnt"]
        base = self.header["sect_off"]
        self.sections = {}
        for i in range(cnt):
            typ, flags, off, ln, al, crc = _SEC.unpack_from(self._m, base + i * _SEC.size)
            self.sections[typ] = (off, ln)
        self._names, self._ent, self._pool = self._parse_nam()

    def _parse_nam(self):
        if SEC_NAM not in self.sections:
            raise ValueError("TG4 缺 NAM 名字表。修法：用本仓库 write_tg4 重写；"
                             "没有名字表就无法按名取张量")
        off, ln = self.sections[SEC_NAM]
        rows = json.loads(self._m[off:off + ln].decode("utf-8"))
        names, ent = [], {}
        for r in rows:
            nm = r["name"]
            names.append(nm)
            ent[nm] = dict(off=int(r["off"]), dtype=int(r["dtype"]),
                           shape=tuple(int(x) for x in r["shape"]))
        return names, ent, None

    def names(self):
        return list(self._names)

    def get(self, name):
        if name not in self._ent:
            raise KeyError("TG4 里没有张量 %r。修法：用 names() 列出可用名字" % name)
        e = self._ent[name]
        base = self.sections[SEC_WGT][0]
        n = int(np.prod(e["shape"])) if e["shape"] else 1
        buf = self._m[base + e["off"]: base + e["off"] + n * _ESZ[e["dtype"]]]
        return _dec(buf, e["dtype"], e["shape"])

    def json_section(self, sec):
        if sec not in self.sections:
            return None
        off, ln = self.sections[sec]
        return json.loads(self._m[off:off + ln].decode("utf-8"))

    def close(self):
        try:
            self._m.close()
        finally:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

def write_tg4_stream(path, items, cfg, arc=None, train=None, per_tensor_hash=False):
    """**流式**写 TG4：items 是 (name, ndarray) 的迭代器，写一个释放一个。

    为什么需要：3B 模型 983 个张量合计 fp32 约 12 GB，而 write_tg4 要求全部同时在内存
    （实测在 32 GB 机器上会把进程拖死）。这里改成**先流式写 WGT，再写 NAM 与分区表**，
    峰值内存 = 单张量（embedding ≈ 67 MB）。

    布局与 write_tg4 兼容（读端不需要区分）：
        128B 头占位 → WGT（边写边记偏移/哈希）→ NAM/ARC/TRAIN/HASH → section table → 回填头
    """
    WGT_START = HEADER_BYTES
    meta, hashes = [], {}
    with open(path, "wb") as f:
        f.write(b"\x00" * HEADER_BYTES)
        pos = HEADER_BYTES
        for name, arr in items:
            a = np.ascontiguousarray(arr, dtype=np.float32)
            pad = (-pos) % ALIGN
            if pad:
                f.write(b"\x00" * pad)
                pos += pad
            blob = _enc(a, cfg.get("weight_dtype", DT_BF16))
            f.write(blob)
            meta.append({"name": name, "off": pos - WGT_START,
                         "dtype": cfg.get("weight_dtype", DT_BF16),
                         "shape": list(a.shape)})
            if per_tensor_hash:
                hashes[name] = hashlib.sha256(blob).digest()
            pos += len(blob)
        wgt_len = pos - WGT_START
        # 之后的分区；最后写表
        payloads = [(SEC_NAM, json.dumps(
            [{"name": m["name"], "off": m["off"], "dtype": m["dtype"],
              "shape": m["shape"]} for m in meta], ensure_ascii=False).encode("utf-8"))]
        if arc is not None:
            payloads.append((SEC_ARC, json.dumps(arc, ensure_ascii=False).encode("utf-8")))
        if train is not None:
            payloads.append((SEC_TRAIN, json.dumps(train, ensure_ascii=False).encode("utf-8")))
        if per_tensor_hash:
            payloads.append((SEC_HASH, b"".join(hashes[m["name"]] for m in meta)))
        entries = [(SEC_WGT, 0, WGT_START, wgt_len, ALIGN, 0)]
        pos = HEADER_BYTES + wgt_len
        for typ, data in payloads:
            pad = (-pos) % ALIGN
            if pad:
                f.write(b"\x00" * pad)
                pos += pad
            entries.append((typ, 0, pos, len(data), ALIGN, 0))
            f.write(data)
            pos += len(data)
        pad = (-pos) % ALIGN
        if pad:
            f.write(b"\x00" * pad)
            pos += pad
        tbl_off = pos
        tbl = bytearray()
        for e in entries:
            tbl += _SEC.pack(*e)
        f.write(bytes(tbl))
        h = dict(cfg)
        h["tok_off"], h["tok_len"] = 0, 0
        h["sect_off"], h["sect_cnt"] = tbl_off, len(entries)
        f.seek(0)
        f.write(_pack_header(h))
    return path
