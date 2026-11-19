# -*- coding: utf-8 -*-
"""远端环境体检：版本 / 文件指纹 / GPU / 已有产物时间戳。

只读，不改任何东西。跑法（本地）：
    python lab/rent/x.py put lab/rent/probe_remote.py D:\orv_bench\probe_remote.py
    python lab/rent/x.py py D:\orv_bench\probe_remote.py
"""
import hashlib
import json
import os
import subprocess
import sys

WORK = r"D:\orv_bench"
TRAIN = r"D:\orv_pip_0962\Lib\site-packages\tgai_go\train"
FILES = [
    r"D:\orv_bench\prof_2048.json",
    r"D:\orv_bench\prof_512.json",
    r"D:\orv_bench\prof_smoke.json",
    r"D:\orv_bench\v3_pyt_train.py",
    r"D:\orv_bench\model_v3.py",
    os.path.join(TRAIN, "cuda.py"),
    os.path.join(TRAIN, "v3_pretrain.py"),
    os.path.join(TRAIN, "int8_adamw_gpu.py"),
    os.path.join(TRAIN, "v3_csa.py"),
    os.path.join(TRAIN, "v3_ops.py"),
    os.path.join(TRAIN, "v3_moe.py"),
]


def md5(path, cap=None):
    h = hashlib.md5()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
            if cap and n >= cap:
                break
    return h.hexdigest()


def main():
    print("== python ==")
    print(sys.version.replace("\n", " "))
    print("executable:", sys.executable)

    import tgai_go.train as T
    ver = getattr(T, "__version__", None)
    if ver is None:
        try:
            import importlib.metadata as md
            ver = md.version("tgai-orv")
        except Exception as ex:                              # noqa: BLE001
            ver = "unknown(%r)" % (ex,)
    print("tgai_go.train version:", ver)

    print("\n== torch ==")
    p = subprocess.run([r"C:\Python311\python.exe", "-c",
                        "import torch;print(torch.__version__, torch.version.cuda,"
                        " torch.cuda.get_device_name(0))"],
                       capture_output=True, text=True)
    print((p.stdout or "").strip() or (p.stderr or "")[-500:])

    print("\n== gpu ==")
    p = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,"
                        "clocks.max.sm,clocks.sm,temperature.gpu,power.draw",
                        "--format=csv"], capture_output=True, text=True)
    print((p.stdout or "").strip() or (p.stderr or "")[-300:])

    print("\n== files ==")
    for f in FILES:
        if not os.path.exists(f):
            print("  MISSING %s" % f)
            continue
        st = os.stat(f)
        print("  %-64s %10d B  %s  md5=%s"
              % (f, st.st_size,
                 __import__("time").strftime("%Y-%m-%d %H:%M:%S",
                                             __import__("time").localtime(st.st_mtime)),
                 md5(f)))

    print("\n== data dir ==")
    for d in (r"D:\orv_bench\data2048_v96000", r"D:\orv_bench\data2048_v4096"):
        if not os.path.isdir(d):
            print("  MISSING %s" % d)
            continue
        items = sorted(os.listdir(d))
        tot = sum(os.path.getsize(os.path.join(d, x)) for x in items
                  if os.path.isfile(os.path.join(d, x)))
        print("  %s  files=%d  bytes=%d  %s" % (d, len(items), tot, items[:6]))
        mp = os.path.join(d, "meta.json")
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                try:
                    meta = json.load(f)
                except Exception:                             # noqa: BLE001
                    meta = "<unparsable>"
            print("     meta.json:", json.dumps(meta, ensure_ascii=False)[:400])

    print("\n== cwd ==")
    print(os.getcwd())


if __name__ == "__main__":
    main()
