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


def _run_two_ranks(tmp_path, job, mode="basic"):
    outdir = str(tmp_path / "dp_out")
    procs = []
    for rank in (0, 1):
        procs.append(subprocess.Popen(
            [sys.executable, "-X", "utf8", RUNNER, str(rank), "2", job, outdir, mode],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", cwd=REPO))
    outs = []
    for p in procs:
        out, _ = p.communicate(timeout=180)
        outs.append((p.returncode, out))
    bad = [(rc, out) for rc, out in outs]
    if any(rc != 0 for rc, _ in outs):
        # ★ 两侧输出都带上：DP 卡死时只有一侧会报超时，另一侧往往已经异常退出，
        #   只贴一侧等于把真正的根因丢掉（2026-11-13 踩过）。
        detail = "\n".join("--- rank%d rc=%d ---\n%s" % (i, rc, (out or "")[-1200:])
                           for i, (rc, out) in enumerate(bad))
        raise AssertionError("DP 子进程失败：\n%s" % detail)
    res = []
    for rank in (0, 1):
        with open(os.path.join(outdir, "rank%d_%s.json" % (rank, mode)), encoding="utf-8") as f:
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


def test_missing_grad_on_one_rank_does_not_deadlock(tmp_path):
    """**某个 rank 缺一层梯度**时必须正常完成，且缺失方按 0 参与平均。

    ## 为什么这条必须存在

    MoE 里某层专家在这个 rank 上没被路由到、或某层梯度先被释放，都会出现
    "只有一边有 grad"。旧实现按"有梯度的条目"拼缓冲：各 rank 的条目数不同 ⇒
    字节总数不同 ⇒ 共享内存名字（`tgai_dp_<job>_<need>`）不同 ⇒
    **rank0 创建 A、rank1 等 B，永远等不到**。表现为训练卡死，且报错信息会误导
    （说"rank0 没创建共享内存"）。

    期望：a 的梯度两边都有 ⇒ 平均 (1+3)/2 = 2；b 只有 rank0 有 [5,5,5] ⇒ 平均 (5+0)/2 = 2.5。
    """
    job = "pytest_dp_%s" % uuid.uuid4().hex[:8]
    try:
        res = _run_two_ranks(tmp_path, job, mode="missing")
    finally:
        for path in glob.glob(os.path.join(os.path.dirname(str(tmp_path)), "**",
                                           "tgai_dp_%s*" % job), recursive=True):
            shutil.rmtree(path, ignore_errors=True)

    for r in res:
        assert r["a_grad"] == [2.0, 2.0, 2.0], (
            "rank%d 的 a 梯度应是 (1+3)/2=2，实际 %s" % (r["rank"], r["a_grad"]))
        assert r["b_grad"] == [2.5, 2.5, 2.5], (
            "rank%d 的 b 梯度应是 (5+0)/2=2.5（缺梯度按 0 补齐），实际 %s"
            % (r["rank"], r["b_grad"]))
        # broadcast 也要真的写回：rank1 的初值是 2/6，广播后必须变成 rank0 的 1/5
        assert r["a_data"] == [1.0, 1.0, 1.0], "rank%d 的 a 权重未被广播覆盖: %s" % (r["rank"], r["a_data"])
        assert r["b_data"] == [5.0, 5.0, 5.0], "rank%d 的 b 权重未被广播覆盖: %s" % (r["rank"], r["b_data"])
