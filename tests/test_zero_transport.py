# -*- coding: utf-8 -*-
"""ZeRO 传输层的**真 2 进程** CPU 测试（不需要显卡）。

单进程测不出来的东西全在这里：共享内存名字/槽复用、文件屏障配对、``all_gather``
与 ``reduce_scatter`` 的分段覆盖、以及「ZeRO-2 的分布式 gnorm 与不分片参考路径
**逐位相同**」这条硬判据。

做法：拉起 ``tests/zero_worker.py`` 两次（同一个 ``TGAI_DP_JOB`` / ``TGAI_DP_DIR``），
两个进程用真的 :class:`tgai_go.train.dist.DataParallel`（host 后端）+ 假运行时
（numpy 实现设备缓冲），跑完把结果以 JSON 打出来，主测试进程对拍。
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORKER = os.path.join(HERE, "zero_worker.py")
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, HERE)

from tgai_go.train import zero as Z            # noqa: E402
from zero_fake import NBuf, reference_ssq_total   # noqa: E402


def _spawn(rank, world, job, dp_dir, extra_env=None):
    env = dict(os.environ)
    env.update({
        "TGAI_DP_WORLD": str(world),
        "TGAI_DP_RANK": str(rank),
        "TGAI_DP_JOB": str(job),
        "TGAI_DP_DIR": str(dp_dir),
        "TGAI_DP_TIMEOUT": "30",
        "TG_DP_BACKEND": "host",
        "PYTHONIOENCODING": "utf-8",
    })
    if extra_env:
        env.update(extra_env)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return subprocess.Popen([sys.executable, "-X", "utf8", WORKER,
                             str(rank), str(world), str(job)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, cwd=REPO)


def _collect(procs, timeout=120):
    outs = []
    for p in procs:
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            for q in procs:
                try:
                    q.kill()
                except Exception:                            # noqa: BLE001
                    pass
            raise AssertionError(
                "ZeRO 2 进程测试超时（%ds）—— 多半是 barrier/共享内存没配上对。"
                "修法：单独跑 `python tests/zero_worker.py 0 2 job0` 与 "
                "`python tests/zero_worker.py 1 2 job0` 看卡在哪" % timeout)
        rc = p.returncode
        text = out.decode("utf-8", "replace")
        errs = err.decode("utf-8", "replace")
        if rc != 0:
            raise AssertionError(
                "ZeRO worker rank rc=%d\n--- stdout ---\n%s\n--- stderr ---\n%s"
                % (rc, text[-4000:], errs[-4000:]))
        rows = [ln for ln in text.splitlines() if ln.startswith("ZERO_WORKER_JSON ")]
        assert rows, "worker 没有输出 ZERO_WORKER_JSON：\n%s\n%s" % (text, errs)
        outs.append(json.loads(rows[-1][len("ZERO_WORKER_JSON "):]))
    return outs


@pytest.mark.parametrize("use_register", [True, False])
def test_two_process_zero2_transport_and_bitwise_gnorm(tmp_path, use_register):
    """ZeRO-2 的完整链路：reduce_scatter → 分布式 gnorm → all_gather。

    四条硬判据（任何一条不满足都说明「逐位一致」是假的）：
      1. 本 rank 拥有的梯度 == 跨 rank 平均的对应切片（逐位）；
      2. gnorm / gscale == 不分片参考路径（**逐位**，不是近似）；
      3. defer 路径算完范数后非本 rank 梯度被释放（ZeRO-2 的显存收益兑现）；
      4. all_gather 后每个 rank 的参数表逐位一致。

    ``use_register=False`` 走「共享内存注册成页锁定失败」的退化路径
    （pinned + copyto）—— 两条路径都必须逐位同值，否则说明某条路径的偏移写错了。
    """
    job = "zt%s_%s" % ("r" if use_register else "n", os.getpid())
    dp_dir = str(tmp_path)
    envx = {} if use_register else {"ZERO_TEST_NOREG": "1"}
    procs = [_spawn(r, 2, job, dp_dir, envx) for r in range(2)]
    rows = _collect(procs)
    assert len(rows) == 2
    for row in rows:
        assert row["owned_ok"], "rank%d 拥有的梯度与参考不一致（maxdiff=%g）" % (
            row["rank"], row["owned_maxdiff"])
        assert row["kept_unowned_before_norm"], \
            "rank%d 在算 gnorm 之前就把非本 rank 梯度删了（gnorm 会与参考不同）" % row["rank"]
        assert row["gnorm_exact"], (
            "rank%d 的 gnorm 与不分片参考**不逐位相同**：%.17g vs %.17g "
            "(gscale %.17g vs %.17g)"
            % (row["rank"], row["gnorm"], row["ref_norm"],
               row["gscale"], row["ref_scale"]))
        assert row["freed_after_norm"], \
            "rank%d 算完范数后没有释放非本 rank 梯度（ZeRO-2 白做了）" % row["rank"]
        assert row["gather_ok"], "rank%d 的 all_gather 没有把参数收回成逐位一致" % row["rank"]
        assert row["n_owned"] > 0
    # 两个 rank 的 gnorm 必须一模一样（否则裁剪阈值不同 ⇒ 梯度不同步）
    assert rows[0]["gnorm"] == rows[1]["gnorm"], (
        "两个 rank 的 gnorm 不同：%.17g vs %.17g"
        % (rows[0]["gnorm"], rows[1]["gnorm"]))
    # 统计口径：reduce_scatter 单向 = 每 rank (w-1)×own；all_gather 同
    for row in rows:
        st = row["stats"]
        assert st["reduce_scatter_send_elems"] > 0
        assert st["gather_send_elems"] > 0
        assert st["gather_send_elems"] == st["gather_recv_elems"]
        assert st["reduce_scatter_send_elems"] == st["reduce_scatter_recv_elems"]


def test_transport_worker_uses_only_public_dataparallel_api():
    """回归锁：``zero.py`` 不许依赖 dist.py 里别的新增方法（那些是其他任务的写作用域）。

    只允许：world / rank / job / timeout / _start / _barrier / _shm / broadcast /
    all_reduce_grads / close —— 全是 1.0.15 已有的（HEAD）接口。
    """
    import io
    src = io.open(os.path.join(REPO, "python", "tgai_go", "train", "zero.py"),
                  encoding="utf-8").read()
    for forbidden in ("all_gather_flat", "reduce_scatter_flat",
                      "all_reduce_array", "_aux_slot", "_nccl_rt"):
        assert forbidden not in src, (
            "zero.py 里出现了 dist.py 的新增方法 %r —— 那属于另一个任务的写作用域，"
            "ZeRO 必须自包含" % (forbidden,))
