# -*- coding: utf-8 -*-
"""远端 train/ 目录全量 md5 清单 → D:\\orv_bench\\remote_md5.json（只读）。

配合本地 cmp_md5.py 校验「远端装的就是本地 HEAD」。
"""
import hashlib
import json
import os

TRAIN = r"D:\orv_pip_0962\Lib\site-packages\tgai_go\train"
OUT = r"D:\orv_bench\remote_md5.json"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    rec = {}
    for name in sorted(os.listdir(TRAIN)):
        p = os.path.join(TRAIN, name)
        if not os.path.isfile(p) or not name.endswith(".py"):
            continue
        st = os.stat(p)
        rec[name] = {"md5": md5(p), "size": st.st_size, "mtime": int(st.st_mtime)}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1, sort_keys=True)
    print("wrote %s  files=%d" % (OUT, len(rec)))


if __name__ == "__main__":
    main()
