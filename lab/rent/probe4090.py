# -*- coding: utf-8 -*-
"""4090 实例环境探测。"""
import shutil
import subprocess
import sys

print("python", sys.version.split()[0], sys.executable)

try:
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
          "cuda", torch.version.cuda)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("dev", torch.cuda.get_device_name(0),
              "%.1f GiB" % (p.total_memory / 2 ** 30), "sm_%d%d" % p.major_minor if False else "sm_%d%d" % (p.major, p.minor))
except Exception as e:                                    # noqa: BLE001
    print("torch 不可用:", type(e).__name__, e)

for mod in ("numpy", "paramiko"):
    try:
        m = __import__(mod)
        print(mod, getattr(m, "__version__", "?"))
    except Exception:                                     # noqa: BLE001
        print(mod, "缺失")

print("pip", shutil.which("pip") or "(用 python -m pip)")
for cmd in (["df", "-h", "/root"], ["df", "-h", "/root/autodl-tmp"]):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
        if lines:
            print(" | ".join(lines[-1].split()))
    except Exception:                                     # noqa: BLE001
        pass
