# -*- coding: utf-8 -*-
"""看一眼 GGUF 里的元数据与张量清单（为 Qwen2.5-0.5B 前向对齐做准备）。"""
import sys

GGUF = r"D:\DSH-Workspace\_gguf_test\qwen2.5-0.5b-fp16.gguf"

try:
    import gguf
    print("用 python-gguf 读取")
    rd = gguf.GGUFReader(GGUF)
    print("=== 元数据（关键项）===")
    for k in rd.fields:
        if any(s in k for s in ("block_count", "embedding_length", "head_count",
                                "context_length", "rope", "ffn", "file_type",
                                "name", "vocab_size", "attention")):
            try:
                f = rd.fields[k]
                v = f.contents()
                if isinstance(v, (bytes, bytearray)):
                    v = bytes(v).decode("utf-8", "replace")[:40]
                print("  %-46s %s" % (k, v))
            except Exception as exc:  # noqa: BLE001
                print("  %-46s <%s>" % (k, exc))
    print("=== 张量（前 24 个）===")
    for t in rd.tensors[:24]:
        print("  %-44s %-10s %s" % (t.name, t.tensor_type.name, list(t.shape)))
    print("张量总数", len(rd.tensors))
except ImportError:
    print("没有 python-gguf，改为手工解析头部")
    import struct
    with open(GGUF, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", magic
        ver = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        print("版本", ver, "张量数", n_tensors, "kv 数", n_kv)

        def rd_str():
            ln = struct.unpack("<Q", f.read(8))[0]
            return f.read(ln).decode("utf-8", "replace")

        TYPES = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

        def rd_val(t):
            if t == 8:
                return rd_str()
            if t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                return [rd_val(et) for _ in range(n)]
            if t in TYPES:
                fmt = TYPES[t]
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
            raise ValueError("未知类型 %d" % t)

        kvs = {}
        for _ in range(n_kv):
            key = rd_str()
            vt = struct.unpack("<I", f.read(4))[0]
            kvs[key] = rd_val(vt)
        for k, v in kvs.items():
            if any(s in k for s in ("block_count", "embedding_length", "head_count",
                                    "context_length", "rope", "ffn", "file_type")):
                print("  %-46s %s" % (k, v if not isinstance(v, list) else "<list %d>" % len(v)))
        print("=== 张量（前 24 个）===")
        for _ in range(min(24, n_tensors)):
            name = rd_str()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            tt = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            print("  %-44s type=%-3d %s" % (name, tt, dims))
