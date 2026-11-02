# -*- coding: utf-8 -*-
"""P2P 全归约自检：同一台机器上跑 world 个 rank（**可以只有 1 张卡**）。

单卡也能验证协议正确性：CUDA IPC 句柄交换、槽位编号 (r*n+i)、按 rank 顺序求和、平均。
跨卡带宽/延迟只能在真·多卡上量，这里不声称。

用法：
    python tools/dp_selftest.py                 # 自动起 2 个 rank（同卡）
    TGAI_DP_P2P=1 python tools/dp_selftest.py
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))


def _child(rank, world, outdir, job):
    os.environ["TGAI_DP_WORLD"] = str(world)
    os.environ["TGAI_DP_RANK"] = str(rank)
    os.environ["TGAI_DP_P2P"] = "1"
    os.environ["TGAI_DP_JOB"] = job

    import numpy as np
    import tgai_go.train as T
    from tgai_go.train.dist_p2p import P2PAllReduce

    T.set_default_device("cuda")
    n = 4096
    rng = np.random.default_rng(1000 + rank)
    params = []
    vals = []
    for i in range(3):
        p = T.tensor(np.zeros((n,), np.float32), requires_grad=True)
        g = rng.normal(size=(n,)).astype(np.float32) + float(rank)
        p.grad = T.tensor(g)
        params.append(p)
        vals.append(g.astype(np.float64))

    ar = P2PAllReduce()
    if not ar.enabled:
        raise SystemExit("P2P 未启用：确认 TGAI_DP_P2P=1 且 world>1")
    ar.attach(params)
    ar.all_reduce_grads(params)
    T._cuda().sync()
    got = [np.asarray(p.grad.to_numpy(), np.float64) for p in params]
    with open(os.path.join(outdir, "rank%d.json" % rank), "w", encoding="utf-8") as f:
        json.dump({"got": [g.tolist() for g in got],
                   "src": [v.tolist() for v in vals]}, f)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--child", type=int, default=-1)
    ap.add_argument("--outdir", default="")
    ap.add_argument("--job", default="selftest")
    a = ap.parse_args()

    if a.child >= 0:
        return _child(a.child, a.world, a.outdir, a.job)

    outdir = tempfile.mkdtemp(prefix="dp_selftest_")
    procs = []
    for r in range(a.world):
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        procs.append(subprocess.Popen(
            [sys.executable, "-X", "utf8", os.path.abspath(__file__),
             "--child", str(r), "--world", str(a.world),
             "--outdir", outdir, "--job", a.job], env=env))
    rc = [p.wait() for p in procs]
    if any(rc):
        print("RESULT FAIL 子进程退出码 %s（看上面的报错）" % rc)
        return 1

    import numpy as np
    runs = []
    for r in range(a.world):
        with open(os.path.join(outdir, "rank%d.json" % r), encoding="utf-8") as f:
            runs.append(json.load(f))
    worst = 0.0
    for i in range(len(runs[0]["got"])):
        expect = np.mean([np.array(run["src"][i]) for run in runs], axis=0)
        for r, run in enumerate(runs):
            got = np.array(run["got"][i])
            worst = max(worst, float(np.abs(got - expect).max() / (np.abs(expect).max() or 1.0)))
    print("RESULT world=%d 各 rank 梯度 == 平均值  最大相对误差 %.3e" % (a.world, worst))
    print("RESULT 结论：%s（单卡双进程，仅验证协议；跨卡带宽未测）"
          % ("PASS" if worst < 1e-6 else "FAIL"))
    return 0 if worst < 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
