# -*- coding: utf-8 -*-
"""本地 train/*.py 与远端清单对比（读 lab/rent/remote_md5.json）。

用法： python lab/rent/cmp_md5.py
输出： 一致数 / 不一致清单 / 仅单侧存在的文件。
"""
import hashlib
import json
import os
import sys

LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "python", "tgai_go", "train")
REMOTE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote_md5.json")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    local = os.path.abspath(LOCAL)
    with open(REMOTE, encoding="utf-8") as f:
        rem = json.load(f)
    same, diff, only_l, only_r = [], [], [], []
    names = sorted(set(os.listdir(local)) | set(rem))
    for n in names:
        lp = os.path.join(local, n)
        lok = n.endswith(".py") and os.path.isfile(lp)
        rok = n in rem
        if lok and rok:
            (same if md5(lp) == rem[n]["md5"] else diff).append(n)
        elif lok:
            only_l.append(n)
        elif rok:
            only_r.append(n)
    print("local :", local)
    print("same  : %d" % len(same))
    print("diff  : %d  %s" % (len(diff), diff))
    print("only local : %d  %s" % (len(only_l), only_l))
    print("only remote: %d  %s" % (len(only_r), only_r))
    return 0 if not diff else 1


if __name__ == "__main__":
    sys.exit(main())
