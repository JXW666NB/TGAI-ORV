# -*- coding: utf-8 -*-
"""环境探针（纯 Python，不经过 shell）：数据集清单 / 包文件指纹 / 版本。

用途：
  * 确认 data2048_v4096 与 data2048_v96000 的分片结构是否可比（分片数/形状/字节数）；
  * 确认远端安装的 tgai_go 与本地源码树**是否同一份**（md5）—— 报告里要说清口径；
  * 打印 torch 版本。
"""
import hashlib
import json
import os

ROOT = r"D:\orv_bench"
PKG = r"D:\orv_pip_0962\Lib\site-packages\tgai_go\train"


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _npz_shape(path):
    try:
        import numpy as np
        with np.load(path, mmap_mode="r") as z:            # 只读头，不整块加载
            out = {}
            for k in z.files:
                a = z[k]
                out[k] = {"shape": list(a.shape), "dtype": str(a.dtype)}
            return out
    except Exception as e:                                  # noqa: BLE001
        return {"err": repr(e)[:200]}


def _dir(p):
    rows = []
    for name in sorted(os.listdir(p)):
        fp = os.path.join(p, name)
        if os.path.isfile(fp):
            rows.append({"name": name, "bytes": os.path.getsize(fp)})
    return rows


print("== 数据集 ==", flush=True)
for d in ("data2048", "data2048_v4096", "data2048_v96000"):
    p = os.path.join(ROOT, d)
    rows = _dir(p)
    tot = sum(r["bytes"] for r in rows)
    print("%-18s files=%d total=%.1f MiB" % (d, len(rows), tot / (1 << 20)), flush=True)
    for r in rows[:3]:
        print("    %-24s %8.2f MiB  %s"
              % (r["name"], r["bytes"] / (1 << 20),
                 json.dumps(_npz_shape(os.path.join(p, r["name"])), ensure_ascii=False)),
              flush=True)
    if rows:
        print("    e.g. %s" % (rows[0]["name"],), flush=True)

print("== 包文件 md5（远端 site-packages） ==", flush=True)
for name in ("__init__.py", "cuda.py", "v3_pretrain.py", "v3_train_cli.py",
             "int8_adamw_gpu.py", "v3_ops.py", "amp.py", "amp_api.py"):
    p = os.path.join(PKG, name)
    if os.path.exists(p):
        print("  %-18s %10d  %s" % (name, os.path.getsize(p), _md5(p)), flush=True)
    else:
        print("  %-18s MISSING" % name, flush=True)

print("== 版本 ==", flush=True)
import tgai_go
print("  tgai_go version:", getattr(tgai_go, "__version__", "n/a"), flush=True)
try:
    import torch
    print("  torch:", torch.__version__, "cuda:", torch.version.cuda, flush=True)
except Exception as e:                                      # noqa: BLE001
    print("  torch: n/a (%r)" % (e,), flush=True)
