# -*- coding: utf-8 -*-
"""判定一个 `.TG` 到底是哪种"v4"。

## 背景：仓库里存在两套"v4"，不能混

1. **`pack.py::TGWriter.write_stream(version=4)`** —— 仍是 **v3 布局**（定长 128B 头 +
   FNV 权重表），只是头里的 `version` 字段写成 4。ORV 的训练检查点走这条。
2. **`tg4.py`** —— 真正的新容器：magic 是 `0x00344754`（`"TG4\\0"`）、
   128B 头 + **section table** + `NAM` 名字表。**全仓库目前无人 import**。

两者的 magic 不同，所以可以一句话分辨；但"v3 布局 + version=4"能不能被 v3 读者读通，
必须**实测**（`read_tg.py` 能不能解析出来），不能假设。

用法：`python probe_tg_layout.py <file.tg> [...]`
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

TGAI_MAGIC = 0x49414754        # b"TGAI"  —— v3 家族
TG4_MAGIC = 0x00344754         # b"TG4\0" —— tg4.py 的新容器


def classify(path):
    with open(path, "rb") as fh:
        head = fh.read(128)
    if len(head) < 128:
        return dict(kind="太短", )
    magic, ver = struct.unpack_from("<II", head, 0)
    out = dict(path=path, size=os.path.getsize(path), magic=magic, version=ver)
    if magic == TG4_MAGIC:
        out["kind"] = "TG4 新容器（section table）"
    elif magic == TGAI_MAGIC:
        # 布局（来自 read_tg.py）：magic@0 version@4 wdt@8 n_layers@12 d_model@16 n_heads@20
        wdt, n_layers, d_model, n_heads = struct.unpack_from("<IIII", head, 8)
        out["kind"] = "v3 家族（定长头 + FNV 表）"
        out["weight_dtype"] = wdt
        out["n_layers"], out["d_model"], out["n_heads"] = n_layers, d_model, n_heads
    else:
        out["kind"] = "未知 magic"
    return out


def try_read_tg(path):
    try:
        from tgai_go.read_tg import read_tg
    except Exception as exc:                      # noqa: BLE001
        return "无法 import read_tg: %s" % exc
    try:
        info = read_tg(path)
    except Exception as exc:                      # noqa: BLE001
        return "read_tg 失败: %s: %s" % (type(exc).__name__, str(exc)[:160])
    h = info.get("header", {})
    n_t = len(info.get("tensors", []) or info.get("items", []) or [])
    return "read_tg OK：version=%s layers=%s d_model=%s，张量 %d 个" % (
        h.get("version"), h.get("n_layers"), h.get("d_model"), n_t)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for path in sys.argv[1:]:
        print("=" * 90)
        print(path)
        try:
            c = classify(path)
        except OSError as exc:
            print("  打不开: %s" % exc)
            continue
        for k in ("size", "magic", "version", "kind", "n_layers", "d_model", "n_heads"):
            if k in c:
                v = c[k]
                if k == "size":
                    v = "%.1f MB" % (v / 1e6)
                elif k == "magic":
                    v = "%#x" % v
                print("  %-10s %s" % (k, v))
        print("  %-10s %s" % ("解析", try_read_tg(path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
