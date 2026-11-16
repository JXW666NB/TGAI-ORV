# -*- coding: utf-8 -*-
"""判定「1.5B 后段变慢」是不是主机 CPU 热降频。

依据：实测 CPU 占用 ≈ 100%（墙钟 ≈ CPU 时间），GPU 在空转；
分配次数/内核数逐 step 恒定；但步时从 0.55 s 爬到 0.96 s。

做法：每步之间强制 sleep N 秒让 CPU 降温。若慢步消失 => 热降频成立。
对照组：sleep 0（即普通运行）。
"""
import argparse
import sys
import time

import tgai_go.train as T
from tgai_go.train import v3_pretrain as P

T.set_default_device("cuda")
rt = T._cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--d_ff", type=int, default=3328)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cool", type=float, default=2.0, help="每步之间冷却秒数")
    a = ap.parse_args()

    from tgai_go.train import optim as _om
    AW = getattr(_om, "AdamW")
    ostep = AW.step
    st = {"prev": None, "rows": [], "slept": False}

    def step(self, *ar, **kw):
        now = time.perf_counter()
        if st["prev"] is not None:
            st["rows"].append(now - st["prev"])
            if a.cool > 0:
                # 冷却：让主机 CPU 降温，检验是否热降频
                time.sleep(a.cool)
        st["prev"] = time.perf_counter()
        return ostep(self, *ar, **kw)

    AW.step = step

    sys.argv = ["v3_pretrain",
                "--data", r"D:\TGAI-V3-1.5B\01_corpus\packed512\s512",
                "--vocab", "96000", "--seq", "512", "--d", "1024",
                "--layers", str(a.layers), "--E", "4", "--n_q", "16",
                "--d_ff", str(a.d_ff), "--top_k", "2", "--csa", "4",
                "--steps", str(a.steps), "--log-every", "100", "--no-final-save"]
    st["prev"] = time.perf_counter()
    try:
        P.main()
    except SystemExit:
        pass

    print("\n冷却 %.1f s/步 时各步耗时：" % a.cool)
    print("  " + " ".join("%.3f" % t for t in st["rows"]))
    if st["rows"]:
        head = st["rows"][:3]
        tail = st["rows"][-3:]
        print("  前3步均值 %.3f s   后3步均值 %.3f s   比值 %.2fx"
              % (sum(head) / len(head), sum(tail) / len(tail),
                 (sum(tail) / len(tail)) / (sum(head) / len(head))))


if __name__ == "__main__":
    main()
