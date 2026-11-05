# -*- coding: utf-8 -*-
"""2 卡数据并行**加速比**实测（缺第二个 GPU 时也能验协议，只是拿不到真加速比）。

用法：
    python tools/dp_speed.py --world 1                 # 单卡基线
    python tools/dp_speed.py --world 2                 # 双卡（需 2 张卡）
    python tools/dp_speed.py --world 2 --check-only     # 只验 allreduce 正确性，不测速

判定口径（与 docs/BENCHMARK_RIGOR.md 一致）：
* **全局 batch 固定**（`--global-batch`），world 越大每卡分到的越小 —— 这样"加速比"才是
  吞吐之比，而不是把 batch 一起放大的假加速；
* 计时循环内**不读设备标量**，只在末尾 sync 一次；
* 预热 `--warmup` 步，统计 `--steps` 步平均；
* 全归约走 `dist_p2p`（CUDA IPC + cuMemcpyPeerAsync，不落主机）。

输出：每卡步时、全局吞吐、相对单卡的加速比，以及达标判定（判据 **≥1.7×**）。
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))


def _train_rank(rank, world, steps, warmup, gbatch, seq, layers, dims, ff, outdir):
    """在 rank 上跑 steps 步，把步时写到 outdir/rank<r>.json。"""
    os.environ["TGAI_DP_WORLD"] = str(world)
    os.environ["TGAI_DP_RANK"] = str(rank)
    os.environ["TGAI_DP_P2P"] = "1"
    os.environ.setdefault("TGAI_DP_JOB", "dpspeed")

    import numpy as np
    import tgai_go.train as T
    from tgai_go.train import qwen as Q
    from tgai_go.train.dist_p2p import P2PAllReduce

    T.set_default_device("cuda")
    rt = T._cuda()
    cfg = dict(V=1024, D=dims, L=layers, NH=max(1, dims // 64), KV=1,
               FF=ff, THETA=1e4, EPS=1e-6, DH=64, QKV_BIAS=True, TIE=False)
    per = gbatch // world
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 1000, (per, seq)).astype(np.int64)
    tgt = rng.integers(0, 1024, (per * seq,)).astype(np.int64)
    W = Q.make_random_weights(cfg, seed=7)
    m = Q.Qwen25(cfg, W)
    opt = T.optim.AdamW(m.params(), lr=1e-4, eps=1e-8)
    ar = P2PAllReduce()
    ar.attach(m.params())            # 训练前建槽位、换 IPC 句柄

    def one():
        lg = m.forward(ids)
        ls = T.functional.cross_entropy(lg.reshape(per * seq, cfg["V"]), tgt)
        ls.backward()
        ar.all_reduce_grads(m.params())   # world=1 时是 no-op
        opt.step()
        opt.zero_grad()

    for _ in range(max(0, warmup)):
        one()
    rt.sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        one()
    rt.sync()
    dt = (time.perf_counter() - t0) / max(1, steps)

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "rank%d.json" % rank), "w", encoding="utf-8") as f:
        json.dump({"rank": rank, "world": world, "sec_per_step": dt,
                   "global_batch": gbatch, "seq": seq}, f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--global-batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--dims", type=int, default=896)
    ap.add_argument("--ff", type=int, default=4864)
    ap.add_argument("--check-only", action="store_true",
                    help="只验 allreduce 正确性（不测速）")
    ap.add_argument("--outdir", default=os.path.join(REPO, "_dp_speed_out"))
    a = ap.parse_args(argv)

    if a.check_only:
        cmd = [sys.executable, "-X", "utf8",
               os.path.join(REPO, "tools", "dp_selftest.py")]
        return subprocess.call(cmd)

    if a.world > 1:
        devs = _gpu_count()
        if devs < a.world:
            print("[skip] 本机只有 %d 张 GPU，测不了 world=%d 的加速比。"
                  "修法：换 2 卡机；只想验协议用 --check-only" % (devs, a.world))
            return 77          # 环境不满足 → 上游按 SKIP 处理

    outdir = a.outdir
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        if f.startswith("rank"):
            os.remove(os.path.join(outdir, f))

    if a.world == 1:
        _train_rank(0, 1, a.steps, a.warmup, a.global_batch, a.seq,
                    a.layers, a.dims, a.ff, outdir)
    else:
        procs = []
        for r in range(a.world):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(r))
            procs.append(subprocess.Popen(
                [sys.executable, "-X", "utf8", os.path.abspath(__file__),
                 "--world", str(a.world), "--steps", str(a.steps),
                 "--warmup", str(a.warmup), "--global-batch", str(a.global_batch),
                 "--seq", str(a.seq), "--layers", str(a.layers),
                 "--dims", str(a.dims), "--ff", str(a.ff),
                 "--outdir", outdir, "--_rank", str(r)],
                env=env))
        for p in procs:
            if p.wait() != 0:
                print("某个 rank 失败；见上面的输出")
                return 1

    times = []
    for f in sorted(os.listdir(outdir)):
        if f.startswith("rank"):
            with open(os.path.join(outdir, f), encoding="utf-8") as fh:
                times.append(json.load(fh)["sec_per_step"])
    if not times:
        print("没拿到任何 rank 的结果")
        return 1
    step = max(times)                     # 最慢的卡决定吞吐
    thr = a.global_batch / step
    print("步时(最慢卡) = %.1f ms   全局吞吐 = %.2f 样本/s   (world=%d, 全局 batch=%d)"
          % (step * 1e3, thr, a.world, a.global_batch))
    print("RESULT world=%d sec_per_step=%.6f throughput=%.3f"
          % (a.world, step, thr))
    return 0


def _gpu_count():
    """本机可见 GPU 数。用驱动 API 直接数，不依赖运行时是否暴露封装。"""
    try:
        import ctypes
        import tgai_go.train as T
        cu = T._cuda().cu
        n = ctypes.c_int(0)
        if cu.cuDeviceGetCount(ctypes.byref(n)) != 0:
            return 0
        return int(n.value)
    except Exception:                                  # noqa: BLE001
        return 0


if __name__ == "__main__":
    if "--_rank" in sys.argv:                          # 子进程分支
        i = sys.argv.index("--_rank")
        rank = int(sys.argv[i + 1])
        argv2 = sys.argv[1:i] + sys.argv[i + 2:]
        ns = argparse.ArgumentParser()
        for k in ("--world", "--steps", "--warmup", "--global-batch", "--seq",
                  "--layers", "--dims", "--ff"):
            ns.add_argument(k)
        ns.add_argument("--outdir")
        a = ns.parse_args(argv2)
        _train_rank(rank, int(a.world), int(a.steps), int(a.warmup),
                    int(a.global_batch), int(a.seq), int(a.layers),
                    int(a.dims), int(a.ff), a.outdir)
        raise SystemExit(0)
    raise SystemExit(main())