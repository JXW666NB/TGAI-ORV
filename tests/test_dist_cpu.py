# -*- coding: utf-8 -*-
"""数据并行（单机多卡）的**正确性**测试：不需要两张卡。

L3 的「2 卡加速比 ≥1.7×」要硬件才能测，但它的**正确性**部分（权重广播、梯度平均）
可以在 CPU 上用两个进程跑出来 —— 因为我们的 DP 不是 NCCL，而是
「共享内存 + 文件屏障」，与设备无关。

为什么值得测：如果梯度没有正确平均（比如只取了 rank0 的梯度、或重复累加了 scale），
训练照样能跑、loss 照样会降，但**等价 batch 是错的**，多卡结果和单卡对不上。
这种错很难靠"看速度"发现。
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import uuid

import pytest

pytest.importorskip("multiprocessing.shared_memory")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(REPO, "tests", "dp_worker.py")   # 别用下划线开头：.gitignore 有 _*.py


def _run_two_ranks(tmp_path, job):
    outdir = str(tmp_path / "dp_out")
    procs = []
    for rank in (0, 1):
        procs.append(subprocess.Popen(
            [sys.executable, "-X", "utf8", RUNNER, str(rank), "2", job, outdir],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", cwd=REPO))
    outs = []
    for p in procs:
        out, _ = p.communicate(timeout=180)
        outs.append((p.returncode, out))
    for rc, out in outs:
        assert rc == 0, "DP 子进程失败 rc=%d\n%s" % (rc, out[-1500:])
    res = []
    for rank in (0, 1):
        with open(os.path.join(outdir, "rank%d.json" % rank), encoding="utf-8") as f:
            res.append(json.load(f))
    return res


def test_broadcast_and_all_reduce_mean(tmp_path):
    job = "pytest_dp_%s" % uuid.uuid4().hex[:8]
    try:
        res = _run_two_ranks(tmp_path, job)
    finally:
        # 清掉共享内存与标记目录，避免污染后续用例
        for path in glob.glob(os.path.join(os.path.dirname(str(tmp_path)), "**",
                                           "tgai_dp_%s*" % job), recursive=True):
            shutil.rmtree(path, ignore_errors=True)

    # 1) broadcast：两个 rank 的权重必须完全等于 rank0 的初值
    for r in res:
        assert r["data"] == [1.0, 1.0, 1.0], "broadcast 后 rank%d 权重是 %s" % (r["rank"], r["data"])

    # 2) all_reduce：默认求平均 ⇒ [1,1,1] 与 [3,3,3] 平均成 [2,2,2]
    for r in res:
        assert r["grad"] == [2.0, 2.0, 2.0], (
            "all_reduce_grads 应是平均值，rank%d 得到 %s（若是 [1,1,1] 则只取了 rank0，"
            "若是 [4,4,4] 则把两个 rank 的梯度相加了）" % (r["rank"], r["grad"]))
