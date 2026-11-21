# -*- coding: utf-8 -*-
"""低精度梯度通信（`TG_DP_GRAD_DTYPE=bf16|fp16`）的**逐位**正确性测试（纯 CPU）。

## 这个功能为什么必须有这些测试

DP 在 PCIe-only 机器上的瓶颈是**字节数**：fp32 梯度 3.16 GiB（848M 参数），
all-reduce 实测 141 ms，而单卡整步只要 141–251 ms ⇒ 「重叠」救不了 DP。
把通信降到 bf16 后字节数减半（1.58 GiB），这才是能过验收线的那条路。
但低精度通信有四种「看起来正常、其实已经错了」的失败方式，全都只能靠 CPU 逐位测：

1. **参数被污染**：cast 写错了位置，`p.grad` 没被覆盖成均值、或者改到了权重上。
   判据：归约结果**逐位等于** `dequant(quant(该 rank 梯度))` 的均值。
2. **量化公式与 CUDA 内核不一致**：主机侧 numpy 的 bf16 舍入必须与
   `cuda.py::cast_f32_to_bf16` 的 RTNE 公式逐位相同，否则 CPU 对拍通过、GPU 上发散。
3. **world=1 被改动**：单卡不涉及归约精度 ⇒ 梯度必须**逐位不变**（验收硬要求）。
4. **字节数没减半 / 均值语义变了**：`comm_rank_bytes` 必须是 fp32 的一半；
   均值仍是 `sum/world`（等价 batch 不变），只改精度。

GPU 路径（NCCL bf16 all-reduce、设备侧 cast 内核、重叠归约的两条传输）用真卡跑，
见 `docs/DP_BF16_GRAD.md` 的实测表；这里把**与设备无关的算术/协议**全部钉住。
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pytest

from tgai_go.train import dist as D
from tgai_go.train import dist_nccl as N
from tgai_go.train import dp_overlap as O

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ══════════════════════════════════════════════════════════════════════
# 0. 子进程 worker（2 rank 的 host 路径，必须真多进程 —— 共享内存 + 文件屏障）
# ══════════════════════════════════════════════════════════════════════
def _worker_main(argv):
    """argv: rank world job outdir mode。结果写 outdir/rank{R}_{mode}.json。"""
    rank, world, job, outdir, mode = argv
    sys.path.insert(0, os.path.join(REPO, "python"))

    class P:
        """CPU 形态的假参数（`.data` 是 ndarray、没有 `device` ⇒ 走 CPU 分支）。"""

        def __init__(self, arr):
            self.data = np.ascontiguousarray(arr, dtype=np.float32)
            self.grad = None

        def numel(self):
            return int(self.data.size)

        @property
        def shape(self):
            return tuple(self.data.shape)

    dtype = "fp32" if mode == "fp32" else "bf16"
    dp = D.DataParallel(world=int(world), rank=int(rank), job=job,
                        timeout=60.0, grad_dtype=dtype)
    out = {"rank": int(rank), "mode": mode, "grad_dtype": dp.grad_dtype,
           "grad_ef": bool(dp.grad_ef)}
    try:
        if mode in ("fp32", "bf16"):
            # 两个 rank 的梯度**不同**：平均后才能看出「有没有真的归约」
            base = np.array([1.1, -2.3, 0.0078125, 1e-3], dtype=np.float32)
            peer = np.array([3.3, 0.5, -0.0078125, 2e-3], dtype=np.float32)
            p = P(base if int(rank) == 0 else peer)
            p.grad = p.data.copy()
            dp.all_reduce_grads([p])
            out["grad"] = [float(x) for x in p.grad]
            out["own"] = [float(x) for x in p.data]
            out["comm_rank_bytes"] = dp.comm_rank_bytes
            out["stats"] = {k: dp.stats()[k] for k in
                            ("grad_dtype", "comm_rank_bytes",
                             "comm_rank_bytes_fp32")}

        elif mode == "bf16_missing":
            # rank1 缺 b 的梯度（MoE 专家没路由到）⇒ 必须按 0 参与、最后补 0
            a = P(np.array([1.1, 2.2], dtype=np.float32))
            b = P(np.array([5.5, 6.6], dtype=np.float32))
            a.grad = np.array([1.1, 2.2], dtype=np.float32)
            b.grad = (np.array([5.5, 6.6], dtype=np.float32)
                      if int(rank) == 0 else None)
            dp.all_reduce_grads([a, b])
            out["a_grad"] = [float(x) for x in a.grad]
            out["b_grad"] = [float(x) for x in b.grad]

        elif mode in ("ef2", "noef2"):
            # 两步同一梯度：EF 把残差带到下一步，**两步平均**应比无 EF 更准
            steps = 2
            g = np.array([1.01171875, 2.03125], dtype=np.float32)
            p = P(g.copy())
            outs = []
            for _ in range(steps):
                p.grad = g.copy()
                dp.all_reduce_grads([p])
                outs.append([float(x) for x in p.grad])
            out["steps"] = outs
        else:
            raise SystemExit("未知 mode: %r" % (mode,))
    finally:
        dp.close()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "rank%d_%s.json" % (int(rank), mode))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--worker":
    _worker_main(sys.argv[2:])
    raise SystemExit(0)


def _run_two_ranks(tmp_path, mode, env_extra=None):
    """起 2 个进程跑 host 路径（与 tests/test_dist_cpu.py 同一套办法）。"""
    pytest.importorskip("multiprocessing.shared_memory")
    job = "pytest_bf16_%s" % uuid.uuid4().hex[:8]
    outdir = str(tmp_path / "bf16_out")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.path.join(REPO, "python")
    for k, v in (env_extra or {}).items():
        env[k] = str(v)
    try:
        procs = []
        for rank in (0, 1):
            procs.append(subprocess.Popen(
                [sys.executable, "-X", "utf8", os.path.abspath(__file__),
                 "--worker", str(rank), "2", job, outdir, mode],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", cwd=REPO, env=env))
        outs = [(p.communicate(timeout=180)[0], p.returncode) for p in procs]
        bad = [(rc, out) for out, rc in outs if rc != 0]
        if bad:
            detail = "\n".join("--- rc=%d ---\n%s" % (rc, (out or "")[-1200:])
                               for rc, out in bad)
            raise AssertionError("低精度 DP 子进程失败：\n%s" % detail)
        res = []
        for rank in (0, 1):
            with open(os.path.join(outdir, "rank%d_%s.json" % (rank, mode)),
                      encoding="utf-8") as fh:
                res.append(json.load(fh))
        return res
    finally:
        for path in glob.glob(os.path.join(os.path.dirname(str(tmp_path)), "**",
                                           "tgai_dp_%s*" % job), recursive=True):
            shutil.rmtree(path, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════
# 1. 环境变量 / 开关
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("raw,want", [
    ("", "fp32"), ("fp32", "fp32"), ("FP32", "fp32"), ("float32", "fp32"),
    ("none", "fp32"), ("off", "fp32"), ("0", "fp32"),
    ("bf16", "bf16"), ("BF16", "bf16"), ("bfloat16", "bf16"), (" bf16 ", "bf16"),
    ("fp16", "fp16"), ("half", "fp16"), ("float16", "fp16"),
])
def test_grad_dtype_aliases(monkeypatch, raw, want):
    monkeypatch.setenv("TG_DP_GRAD_DTYPE", raw)
    assert D.grad_dtype_requested() == want


def test_invalid_grad_dtype_is_loud(monkeypatch):
    """写错一个字母不能静默退回 fp32（那会悄悄多一倍通信字节）。"""
    monkeypatch.setenv("TG_DP_GRAD_DTYPE", "b16")
    with pytest.raises(ValueError) as ei:
        D.grad_dtype_requested()
    msg = str(ei.value)
    assert "b16" in msg and "修法" in msg and "bf16" in msg


def test_grad_ef_env_is_loud(monkeypatch):
    monkeypatch.setenv("TG_DP_GRAD_EF", "yes")
    assert D.grad_ef_requested() is True
    monkeypatch.setenv("TG_DP_GRAD_EF", "nope")
    with pytest.raises(ValueError, match="TG_DP_GRAD_EF"):
        D.grad_ef_requested()


def test_grad_itemsize():
    assert D.grad_itemsize("fp32") == 4
    assert D.grad_itemsize("bf16") == 2
    assert D.grad_itemsize("fp16") == 2
    with pytest.raises(ValueError, match="修法"):
        D.grad_itemsize("int8")


def test_ef_with_fp32_is_refused(monkeypatch):
    """EF 只对低精度有意义：fp32 时开着它说明用户没理解语义，必须响亮拒绝。"""
    monkeypatch.setenv("TG_DP_GRAD_DTYPE", "fp32")
    monkeypatch.setenv("TG_DP_GRAD_EF", "1")
    with pytest.raises(ValueError) as ei:
        D.DataParallel(world=1, rank=0, job="pytest_ef_fp32")
    assert "TG_DP_GRAD_EF" in str(ei.value) and "修法" in str(ei.value)


# ══════════════════════════════════════════════════════════════════════
# 2. 量化公式：主机侧必须与 CUDA 内核逐位一致
# ══════════════════════════════════════════════════════════════════════
def _bf16_ref_bits(a):
    """`cuda.py::cast_f32_to_bf16` 的公式，逐字照抄（独立于被测实现）。"""
    u = np.ascontiguousarray(a, np.float32).reshape(-1).view(np.uint32)
    r = ((u.astype(np.uint64) + np.uint64(0x7FFF)
          + ((u.astype(np.uint64) >> np.uint64(16)) & np.uint64(1)))
         >> np.uint64(16))
    return r.astype(np.uint16)


def test_bf16_bits_match_cuda_kernel_formula():
    rng = np.random.default_rng(0)
    vals = np.concatenate([
        rng.normal(0, 1, 257).astype(np.float32),
        rng.normal(0, 1e-8, 33).astype(np.float32),     # 接近 0（bf16 指数范围与 fp32 同）
        rng.normal(0, 1e8, 33).astype(np.float32),
        np.array([1.0, -1.0, 0.0, -0.0,
                  1.00390625,        # 1 + 2^-8：正好是 bf16 的**舍入中点**（偶舍入 → 1.0）
                  -1.00390625,
                  1.01171875,        # 1 + 3*2^-8：中点（偶舍入 → 1.015625）
                  3.14159265, -2.7182818], np.float32),
    ])
    got = D.bf16_bits_from_f32(vals)
    want = _bf16_ref_bits(vals)
    np.testing.assert_array_equal(got, want)
    # 已知值（人工核对过位模式）
    assert int(D.bf16_bits_from_f32(np.array([1.0], np.float32))[0]) == 0x3F80
    assert float(D.dequantize_from_bits(
        D.bf16_bits_from_f32(np.array([1.00390625], np.float32)), "bf16")[0]) == 1.0
    assert float(D.dequantize_from_bits(
        D.bf16_bits_from_f32(np.array([1.01171875], np.float32)), "bf16")[0]) == 1.015625


def test_bf16_roundtrip_error_bound():
    """bf16 只有 8 位有效数字 ⇒ 相对舍入误差上界 2^-8（2^-9 是 [1,2) 尾段的典型值）。"""
    rng = np.random.default_rng(1)
    v = rng.normal(0, 1, 4096).astype(np.float32)
    q = D.dequantize_from_bits(D.quantize_to_bits(v, "bf16"), "bf16")
    rel = np.abs(q - v) / np.maximum(np.abs(v), 1e-30)
    assert float(rel.max()) <= 2.0 ** -8 + 1e-7, "bf16 相对误差超过 2^-8"
    assert not np.array_equal(q, v), "bf16 量化不可能逐位等于 fp32（否则没在收窄）"


def test_fp16_quantization_matches_numpy():
    rng = np.random.default_rng(2)
    v = rng.normal(0, 1, 2048).astype(np.float32)
    want = v.astype(np.float16).astype(np.float32)
    got = D.dequantize_from_bits(D.quantize_to_bits(v, "fp16"), "fp16")
    np.testing.assert_array_equal(got, want)


# ══════════════════════════════════════════════════════════════════════
# 3. NumpyTransport：cast 路径 + 均值语义（逐位）
# ══════════════════════════════════════════════════════════════════════
def _session(params, transport, world=1, scale=None, bucket_bytes=None,
             zero_grad=None):
    elems = [int(np.prod(tuple(p.shape))) for p in params]
    plan = O.build_plan(elems, bucket_bytes=(1 << 20 if bucket_bytes is None
                                             else bucket_bytes))
    return O.OverlapSession(params, plan, transport, world=world, scale=scale,
                            zero_grad=zero_grad)


def _drive(sess, tp, params, grads):
    """直接驱动会话：模拟 attach 完成 + 反向逐个写 .grad（不依赖 autograd）。"""
    sess._attached = True
    sess._active = True
    sess._pending = [1] * len(params)
    tp.open(sess)
    for i, p in enumerate(params):
        p.grad = np.array(grads[i], np.float32)
        sess.leaf_ready(p)
    return sess.finish()


def _import_tensor():
    import tgai_go.train as T
    return T


def test_numpy_transport_bf16_cast_path_is_quantized():
    """world=1/scale=1：桶里就是本 rank 的量化梯度，写回必须**逐位**等于量化值。"""
    T = _import_tensor()
    vals = [np.array([1.1, -2.3, 0.0078125], np.float32),
            np.array([[1.0, 1.00390625]], np.float32)]
    params = [T.Tensor(np.zeros_like(v), requires_grad=True) for v in vals]
    tp = O.NumpyTransport(grad_dtype="bf16")
    st = _drive(_session(params, tp), tp, params, vals)
    assert st["active"] is True
    assert st["grad_dtype"] == "bf16"
    for p, v in zip(params, vals):
        want = D.dequantize_from_bits(D.quantize_to_bits(v, "bf16"), "bf16")
        np.testing.assert_array_equal(np.asarray(p.grad, np.float32),
                                      want.reshape(tuple(p.shape)))
    total = sum(int(np.prod(tuple(p.shape))) for p in params)
    assert st["comm_bytes"] == total * 2, "通信字节必须是 fp32 的一半"


def test_numpy_transport_bf16_two_rank_mean_bitwise():
    """模拟 2 rank：对端载荷 = 已知数组；均值必须是 (own_q + peer_q) * 0.5，**逐位**。"""
    T = _import_tensor()
    own = np.array([1.1, -2.3, 0.0078125, 3.14159265], np.float32)
    peer = np.array([3.3, 0.5, -0.0078125, -1.2345678], np.float32)
    params = [T.Tensor(np.zeros_like(own), requires_grad=True)]
    peer_q = D.dequantize_from_bits(D.quantize_to_bits(peer, "bf16"), "bf16")

    def reduce_fn(buf, info, sess):
        buf += peer_q                       # 对端实际会收到的量化载荷

    tp = O.NumpyTransport(reduce_fn=reduce_fn, grad_dtype="bf16")
    st = _drive(_session(params, tp, world=2), tp, params, [own])
    own_q = D.dequantize_from_bits(D.quantize_to_bits(own, "bf16"), "bf16")
    want = (own_q + peer_q) * np.float32(0.5)
    np.testing.assert_array_equal(np.asarray(params[0].grad, np.float32), want)
    assert st["comm_bytes"] == own.size * 2


def test_numpy_transport_bf16_missing_grad_is_zero():
    """缺梯度的参数仍按 0 参与集合通信，最后补全 0 梯度（语义与 fp32 路径一致）。"""
    T = _import_tensor()
    a = T.Tensor(np.zeros((2,), np.float32), requires_grad=True)
    b = T.Tensor(np.zeros((3,), np.float32), requires_grad=True)
    tp = O.NumpyTransport(grad_dtype="bf16")

    def zero_grad(p):
        p.grad = np.zeros(tuple(p.shape), np.float32)

    plan = O.build_plan([2, 3], bucket_bytes=1 << 20)
    sess = O.OverlapSession([a, b], plan, tp, world=1, scale=1.0,
                            zero_grad=zero_grad)
    # 直接模拟 attach_graph 的记账结果：a 可达（pending=1），b 不在图里（pending=0，
    # 且要从桶的待齐计数里扣掉、并记成「按 0 参与」）。
    sess._attached = True
    sess._active = True
    sess._pending = [1, 0]
    sess._bucket_left[plan.param_bucket[1]] -= 1
    sess._zero_members.append(1)
    tp.open(sess)
    a.grad = np.array([1.1, -2.3], np.float32)
    sess.leaf_ready(a)
    st = sess.finish()
    assert st["zero_filled"] == 1
    np.testing.assert_array_equal(np.asarray(b.grad, np.float32),
                                  np.zeros((3,), np.float32))


def test_numpy_transport_fp32_unchanged():
    """默认 fp32：桶里就是原始梯度（不量化），与历史行为逐位一致。"""
    T = _import_tensor()
    v = np.array([1.1, -2.3, 3.14159265], np.float32)
    params = [T.Tensor(np.zeros_like(v), requires_grad=True)]
    tp = O.NumpyTransport()
    st = _drive(_session(params, tp), tp, params, [v])
    np.testing.assert_array_equal(np.asarray(params[0].grad, np.float32), v)
    assert st["grad_dtype"] == "fp32"
    assert "comm_bytes" not in st, "fp32 路径不该报低精度字节口径"


# ══════════════════════════════════════════════════════════════════════
# 4. world=1 零变化（验收硬要求）
# ══════════════════════════════════════════════════════════════════════
def test_world_one_is_bitwise_noop_with_bf16(monkeypatch):
    T = _import_tensor()
    monkeypatch.setenv("TG_DP_GRAD_DTYPE", "bf16")

    class _Boom(object):
        def __getattr__(self, name):
            raise AssertionError("world=1 不能碰 CUDA（访问了 %s）" % name)

    monkeypatch.setattr(T, "_cuda", lambda: _Boom())
    dp = D.DataParallel(world=1, rank=0, job="pytest_bf16_noop")
    assert dp.grad_dtype == "bf16"
    x = T.Tensor(np.ones((4,), np.float32), requires_grad=True)
    before = np.array([1.1, -2.3, 3.14159265, 0.0], np.float32)
    x.grad = before.copy()
    assert dp.all_reduce_grads([x]) is None
    np.testing.assert_array_equal(np.asarray(x.grad, np.float32), before)
    dp.close()


# ══════════════════════════════════════════════════════════════════════
# 5. 误差反馈：默认关；开着时自动关掉重叠路径；两步平均更准
# ══════════════════════════════════════════════════════════════════════
def test_ef_disables_overlap_path(monkeypatch):
    monkeypatch.setenv("TG_DP_GRAD_DTYPE", "bf16")
    monkeypatch.setenv("TG_DP_GRAD_EF", "1")
    T = _import_tensor()
    dp = D.DataParallel(world=2, rank=0, job="pytest_ef_ov")
    assert dp.grad_ef is True
    x = T.Tensor(np.ones((2,), np.float32), requires_grad=True)
    # EF 需要不重叠路径 ⇒ begin_overlap 必须返回 None（且不能走到 rendezvous）
    assert dp.begin_overlap([x]) is None
    dp.close()


def test_lowp_disables_overlap_path(monkeypatch):
    """2026-11-21 GPU 实测：bf16 + 反向重叠的 gnorm 偏大 4.6×（非重叠路径逐位通过）。

    在修好之前低精度一律退回不重叠路径 —— 不能因为「loss 看起来正常」就放过去
    （Adam 近似尺度不变，梯度整体放大 4.6× 时 loss 曲线几乎不变，这是最危险的一类 bug）。

    ★ 这里**必须把 `_start()` 打桩**：`begin_overlap()` 会先做 rendezvous，
      而本用例只起 rank0（world=2），真做就是等 rank1 等到超时
      （CI ubuntu 上 4 条腿全红就是这么来的，白等 10 分钟）。
      我们要验的是" dtype 闸门"，与 rendezvous 无关 ⇒ 打桩后判据更纯、也更稳。
    """
    monkeypatch.delenv("TG_DP_GRAD_EF", raising=False)
    T = _import_tensor()
    dp = D.DataParallel(world=2, rank=0, job="pytest_lowp_ov",
                        grad_dtype="bf16")
    monkeypatch.setattr(dp, "_start", lambda: None)     # 不做 rendezvous
    x = T.Tensor(np.ones((2,), np.float32), requires_grad=True)
    assert dp.begin_overlap([x]) is None, "低精度 + 重叠必须自动退回不重叠路径"
    dp.close()


def test_ef_two_step_average_beats_no_ef(tmp_path):
    """同一梯度连跑两步：EF 把第一步的残差带到第二步，**两步平均**误差更小。

    选值 1.01171875 = 1 + 3·2^-8 正好是 bf16 的舍入中点（偶舍入 → 1.015625）。
    无 EF：两步都量化为 1.015625（平均误差 +0.00390625）。
    有 EF：第一步 1.015625，残差 −0.00390625 ⇒ 第二步输入 1.0078125（精确可表示）
    ⇒ 两步平均正好 1.01171875（误差 0）。
    """
    res_ef = _run_two_ranks(tmp_path, "ef2", {"TG_DP_GRAD_EF": "1"})
    res_no = _run_two_ranks(tmp_path, "noef2", {"TG_DP_GRAD_EF": "0"})
    g = np.array([1.01171875, 2.03125], dtype=np.float32)
    for r in res_ef:
        assert r["grad_ef"] is True
    for r in res_no:
        assert r["grad_ef"] is False
    avg_ef = np.mean(np.array(res_ef[0]["steps"], np.float32), axis=0)
    avg_no = np.mean(np.array(res_no[0]["steps"], np.float32), axis=0)
    err_ef = float(np.abs(avg_ef - g).max())
    err_no = float(np.abs(avg_no - g).max())
    assert err_ef < err_no, "EF 的两步平均误差应更小（ef=%g, no_ef=%g）" % (err_ef, err_no)
    # ★ 不能要求"精确 0"（2026-11-21，CI 只在 windows-latest 3.12 上翻车）：
    #   实测残差 0.001953125 = 2^-9，正好是 bf16 在 1.0 附近（ulp = 2^-8）的**半个 ulp**。
    #   也就是说 EF 已经把误差压到量化步长以内，剩下的那半个 ulp 来自
    #   "两步残差相加后的舍入顺序"，跟 numpy/BLAS 构建有关，不同平台会不一样。
    #   判据改成：**不超过 1 个 bf16 ulp**（值落在 [1,2) 时 ulp = 2^-8 = 0.00390625）。
    #   注意别用 `np.spacing(np.float32(1.0))` —— 那是 **float32** 的 ulp（1.19e-7），
    #   比 bf16 小 32768 倍，等于又变回"要求精确相等"。
    #   这条与 CHANGELOG 里"分层等价"那次是同一类教训：把"算法正确"写成"平台无关的浮点恒等"。
    ulp_bf16 = 2.0 ** -8
    assert err_ef <= ulp_bf16, \
        "EF 残差应在 1 个 bf16 ulp 内（实测 %g > %g）" % (err_ef, ulp_bf16)


# ══════════════════════════════════════════════════════════════════════
# 6. host 路径 2 rank：均值正确 + 字节数减半（真多进程）
# ══════════════════════════════════════════════════════════════════════
def _expected_mean(own, peer, dtype):
    return (D.dequantize_from_bits(D.quantize_to_bits(own, dtype), dtype)
            + D.dequantize_from_bits(D.quantize_to_bits(peer, dtype), dtype)) \
        * np.float32(0.5)


OWN = np.array([1.1, -2.3, 0.0078125, 1e-3], dtype=np.float32)
PEER = np.array([3.3, 0.5, -0.0078125, 2e-3], dtype=np.float32)


def test_host_two_rank_bf16_mean_and_bytes(tmp_path):
    res = _run_two_ranks(tmp_path, "bf16")
    want = _expected_mean(OWN, PEER, "bf16")
    for r in res:
        assert r["grad_dtype"] == "bf16"
        np.testing.assert_array_equal(np.array(r["grad"], np.float32), want,
                                      err_msg="rank%d 的 bf16 均值不对" % r["rank"])
        assert r["stats"]["comm_rank_bytes"] == OWN.size * 2, (
            "通信载荷字节必须是 fp32 的一半：%s" % (r["stats"],))
        assert r["stats"]["comm_rank_bytes"] * 2 == r["stats"]["comm_rank_bytes_fp32"]
    # 与 fp32 结果对比：必须有差（否则 bf16 根本没生效），差量在 bf16 精度量级内
    res32 = _run_two_ranks(tmp_path, "fp32")
    want32 = (OWN + PEER) * np.float32(0.5)
    for r in res32:
        np.testing.assert_array_equal(np.array(r["grad"], np.float32), want32)
        assert r["stats"]["comm_rank_bytes"] is None, "fp32 不该填低精度字节口径"
    diff = float(np.abs(want - want32).max())
    assert 0.0 < diff <= 0.01, "bf16 与 fp32 的差应在 bf16 精度量级（实测 %g）" % diff


def test_host_two_rank_bf16_missing_grad(tmp_path):
    """rank1 缺 b 的梯度：b 按 0 参与 ⇒ b = (q(5.5)+0)/2；a 正常平均。"""
    res = _run_two_ranks(tmp_path, "bf16_missing")
    want_b = (D.dequantize_from_bits(D.quantize_to_bits(
        np.array([5.5, 6.6], np.float32), "bf16"), "bf16") * np.float32(0.5))
    for r in res:
        np.testing.assert_array_equal(np.array(r["b_grad"], np.float32), want_b,
                                      err_msg="rank%d 的缺失梯度没按 0 参与" % r["rank"])


# ══════════════════════════════════════════════════════════════════════
# 7. NCCL 类型通道（假库）：低精度必须带 dtype，fp32 路径签名不变
# ══════════════════════════════════════════════════════════════════════
class _FakeLib(object):
    def __init__(self):
        self.events = []
        self.destroyed = 0
        self.aborted = 0

    def init_rank(self, world, rank, uid):
        return 0xDEAD

    def comm_count(self, comm):
        return 2

    def comm_user_rank(self, comm):
        return 0

    def comm_cu_device(self, comm):
        return 0

    def group_start(self):
        self.events.append(("group_start",))

    def group_end(self):
        self.events.append(("group_end",))

    def all_reduce(self, comm, send, recv, count, stream=None):
        self.events.append(("fp32", send, recv, count))

    def all_reduce_typed(self, comm, send, recv, count, dtype, stream=None):
        self.events.append(("typed", send, recv, count, dtype))

    def destroy(self, comm):
        self.destroyed += 1

    def abort(self, comm):
        self.aborted += 1


def _fake_comm():
    lib = _FakeLib()
    comm = N.NcclComm(lib, 2, 0, N.unique_id_from_bytes(bytes(128)))
    return lib, comm


def test_nccl_group_fp32_path_unchanged():
    lib, comm = _fake_comm()
    comm.all_reduce_group([(0x1000, 32), (0x2000, 0)])
    kinds = [e[0] for e in lib.events]
    assert kinds == ["group_start", "fp32", "group_end"]
    assert ("fp32", 0x1000, 0x1000, 32) in lib.events
    comm.close()


@pytest.mark.parametrize("dtype", [N.NCCL_BFLOAT16, N.NCCL_FLOAT16])
def test_nccl_group_lowp_uses_typed_call(dtype):
    lib, comm = _fake_comm()
    comm.all_reduce_group([(0x3000, 64)], dtype=dtype)
    kinds = [e[0] for e in lib.events]
    assert kinds == ["group_start", "typed", "group_end"], (
        "低精度必须走带 ncclDataType_t 的调用，否则 NCCL 按 fp32 读会越界/算错")
    assert ("typed", 0x3000, 0x3000, 64, dtype) in lib.events
    comm.close()


def test_nccl_dtype_constants_are_correct():
    """ncclDataType_t 的枚举值是 ABI 的一部分，写错就是静默算错。"""
    assert N.NCCL_FLOAT16 == 6
    assert N.NCCL_FLOAT32 == 7
    assert N.NCCL_BFLOAT16 == 9
    assert O._nccl_dtype_for("bf16") == 9
    assert O._nccl_dtype_for("fp16") == 6
    assert O._nccl_dtype_for("fp32") is None


# ══════════════════════════════════════════════════════════════════════
# 8. 设备侧打包管线：用**假运行时**在 CPU 上跑真正的 `_all_reduce_lowp_gpu`
#
# 为什么值得写这层假运行时：GPU 路径（pack32 → cast → ncclAllReduce(bf16) →
# dequant → ×1/world → 覆盖写回）占了生产路径的 100%，但它需要两张卡才能跑。
# 这里的假运行时把 5 个内核按**同一算法**在 numpy 上实现，于是偏移/顺序/分块
# 这些最容易错的地方可以在 CI 里逐位钉住；真卡上去只验「带宽与内核真的快」。
# ══════════════════════════════════════════════════════════════════════
class _FBuf(object):
    """假设备缓冲：`ptr` 是登记表里的整数 token，`.np` 是 numpy 视图。"""

    __slots__ = ("np", "n", "shape", "dtype", "nbytes", "ptr")

    def __init__(self, arr, ptr):
        self.np = np.ascontiguousarray(arr).reshape(-1)
        self.n = int(self.np.size)
        self.shape = (self.n,)
        self.dtype = self.np.dtype
        self.nbytes = int(self.np.nbytes)
        self.ptr = int(ptr)


class _FakeRt(object):
    """只实现低精度打包管线用到的那几个原语（内核按同一算法在 numpy 上模拟）。"""

    def __init__(self):
        self._reg = []                      # [(base_ptr, flat_ndarray)]
        self._next = 1 << 20
        self._stream = None
        self.cu = self
        self.kernel_calls = []

    # ---- 缓冲 ----
    def _alloc(self, shape, dtype):
        shape = tuple(shape) if shape is not None else ()
        arr = np.zeros(shape if shape else (1,), dtype=dtype)
        ptr = self._next
        self._next += 1 << 20
        buf = _FBuf(arr, ptr)
        self._reg.append((ptr, buf.np))
        return buf

    def empty(self, shape, dtype=np.float32):
        return self._alloc(shape, dtype)

    def zeros(self, shape, dtype=np.float32):
        return self._alloc(shape, dtype)

    def sub_view(self, buf, shape, elem_offset=0):
        n = 1
        for s in shape:
            n *= int(s)
        off = int(elem_offset)
        view = _FBuf(buf.np[off:off + n], int(buf.ptr) + off * buf.np.itemsize)
        return view

    def _find(self, ptr):
        """ptr → (底层 flat 数组, 元素偏移)。取包含它的**最内层**登记（base 最大）。"""
        ptr = int(ptr)
        best = None
        for base, flat in self._reg:
            if base <= ptr < base + flat.nbytes:
                if best is None or base > best[0]:
                    best = (base, flat)
        if best is None:
            raise AssertionError("假运行时：指针 %d 没有登记（偏移算错了？）" % ptr)
        base, flat = best
        return flat, (ptr - base) // flat.itemsize

    # ---- 内存操作 ----
    def _split_forget_ptr(self, buf):
        pass

    def _mark_submitted(self, ptr):
        pass

    def _vec4_ok(self, n, *bufs):
        return False                        # 只走标量内核，模拟更直白

    def sync(self):
        pass

    def copy_d2d(self, dst, src, nbytes=None, dst_off=0, src_off=0, stream=None):
        nb = int(nbytes if nbytes is not None else src.nbytes)
        darr, doff = self._find(int(dst.ptr) + int(dst_off))
        sarr, soff = self._find(int(src.ptr) + int(src_off))
        assert darr.itemsize == sarr.itemsize, "copy_d2d 的 dtype 不一致"
        n = nb // darr.itemsize
        darr[doff:doff + n] = sarr[soff:soff + n]

    def scale_inplace(self, a, scalar, stream=None):
        arr, off = self._find(int(a.ptr))
        n = int(a.n)
        arr[off:off + n] = arr[off:off + n] * np.float32(scalar)

    def add_(self, x, y, stream=None):
        xarr, xo = self._find(int(x.ptr))
        yarr, yo = self._find(int(y.ptr))
        n = int(x.n)
        assert int(y.n) == n, "add_ 长度不同"
        xarr[xo:xo + n] = xarr[xo:xo + n] + yarr[yo:yo + n]

    def cuMemsetD32Async(self, dst, value, count, stream=None):
        """4 字节字填充（低精度清零的「成对元素」分支用；只支持填 0）。"""
        ptr = int(dst.value if hasattr(dst, "value") else dst)
        nwords = int(count.value if hasattr(count, "value") else count)
        val = int(value.value if hasattr(value, "value") else value)
        assert val == 0, "假运行时只模拟 memset 0"
        arr, off = self._find(ptr)
        arr[off:off + nwords * (4 // arr.itemsize)] = 0

    # ---- 内核 ----
    def _ewise_grid(self, n):
        return (1, 1, 1), (1, 1, 1)

    def launch(self, name, grid, block, args, shared=0, module=None, stream=None):
        vals = [a[0] for a in args]
        self.kernel_calls.append(name)
        if name in ("cast_f32_to_bf16", "cast_f32_to_f16"):
            sptr, dptr, n = vals
            sarr, so = self._find(sptr)
            darr, do = self._find(dptr)
            src = sarr[so:so + n]
            bits = (D.bf16_bits_from_f32(src) if name.endswith("bf16")
                    else src.astype(np.float16).view(np.uint16))
            darr[do:do + n] = bits
        elif name in ("cast_bf16_to_f32", "cast_f16_to_f32"):
            sptr, dptr, n = vals
            sarr, so = self._find(sptr)
            darr, do = self._find(dptr)
            dt = "bf16" if "bf16" in name else "fp16"
            darr[do:do + n] = D.dequantize_from_bits(sarr[so:so + n], dt)
        elif name == "ewise_scale":
            aptr, s, cptr, n = vals
            aarr, ao = self._find(aptr)
            carr, co = self._find(cptr)
            carr[co:co + n] = aarr[ao:ao + n] * np.float32(s)
        elif name == "fused_add_ip":
            xptr, yptr, n = vals
            xarr, xo = self._find(xptr)
            yarr, yo = self._find(yptr)
            xarr[xo:xo + n] = xarr[xo:xo + n] + yarr[yo:yo + n]
        else:
            raise AssertionError("假运行时还不认识内核 %r（新内核要在这里补模拟）" % (name,))


class _FakeComm(object):
    """模拟 2 rank all-reduce：对端 = 本端 × 2（确定性，便于独立算期望值）。

    ★ 必须按 dtype 解释缓冲：低精度时缓冲里是**位模式**，直接在 uint16 上做加法
      会得到垃圾 —— 真 NCCL 的 bf16 sum 也是「升精度求和再收窄」，这里同构模拟。
    """

    def __init__(self, rt):
        self.rt = rt
        self.calls = []                     # [(count, dtype)]

    def all_reduce_group(self, entries, stream=None, dtype=None):
        for ptr, count in entries:
            count = int(count)
            self.calls.append((count, dtype))
            arr, off = self.rt._find(int(ptr))
            seg = arr[off:off + count]
            if dtype == N.NCCL_FLOAT32 or dtype is None:
                seg += seg * np.float32(2.0)
            else:
                dt = "bf16" if dtype == N.NCCL_BFLOAT16 else "fp16"
                x = D.dequantize_from_bits(seg, dt)
                tot = x + x * np.float32(2.0)      # 对端 = 本端 × 2
                seg[:] = D.quantize_to_bits(tot, dt)


class _FakeParam(object):
    __slots__ = ("grad", "shape", "_n")

    def __init__(self, n):
        self._n = int(n)
        self.shape = (self._n,)
        self.grad = None

    def numel(self):
        return self._n


def _fake_dp(monkeypatch, grad_dtype, grad_ef=False):
    T = _import_tensor()
    rt = _FakeRt()
    monkeypatch.setattr(T, "_cuda", lambda: rt)
    dp = D.DataParallel(world=2, rank=0, job="pytest_fake_%s" % grad_dtype,
                        grad_dtype=grad_dtype)
    dp.backend = "nccl"
    dp._nccl = _FakeComm(rt)
    dp.grad_ef = bool(grad_ef)
    return dp, rt


def _entries(rt, vals):
    """造 (p, grad, n) 三元组；grad 是带 `_gpu`（假运行时缓冲）的假张量。"""
    out = []
    for v in vals:
        v = np.ascontiguousarray(v, np.float32)
        p = _FakeParam(v.size)
        buf = rt.empty((v.size,), np.float32)
        buf.np[:] = v
        g = type("_G", (), {"_gpu": buf, "shape": (v.size,)})()
        p.grad = g
        out.append((p, g, int(v.size)))
    return out


@pytest.mark.parametrize("dtype,dt", [("bf16", N.NCCL_BFLOAT16),
                                      ("fp16", N.NCCL_FLOAT16)])
def test_gpu_pipeline_multi_chunk_matches_reference(monkeypatch, dtype, dt):
    """设备侧完整管线（多分块）：包内每个元素都必须恰好走一遍 cast/reduce/写回。"""
    monkeypatch.delenv("TG_DP_TIMING", raising=False)
    monkeypatch.setattr(D, "_lowp_chunk_elems", lambda total: 4)   # 强制 6 个分块
    dp, rt = _fake_dp(monkeypatch, dtype)
    vals = [np.array([1.1, -2.3, 0.0078125, 1e-3, 3.14159265], np.float32),
            np.array([-1.5, 2.25, 0.5, -0.125, 7.0, -3.5, 0.75], np.float32),
            np.arange(11, dtype=np.float32) - 5.0]
    entries = _entries(rt, vals)
    total = sum(e[2] for e in entries)
    dp._all_reduce_lowp(entries, total, None)

    # 假 comm：sum 在低精度里做 ⇒ 期望 = q(3·q(own)) / 2
    for (p, g, n), v in zip(entries, vals):
        own_q = D.dequantize_from_bits(D.quantize_to_bits(v, dtype), dtype)
        three = own_q + own_q * np.float32(2.0)
        want = (D.dequantize_from_bits(D.quantize_to_bits(three, dtype), dtype)
                * np.float32(0.5))
        np.testing.assert_array_equal(g._gpu.np[:n], want,
                                      err_msg="%s 管线的写回值不对" % dtype)
    # 通信协议：每块都必须是低精度 dtype，count 之和 = 元素总数
    assert dp._nccl.calls, "没有发生集合通信"
    assert all(c[1] == dt for c in dp._nccl.calls), "低精度路径必须带 ncclDataType_t"
    assert sum(c[0] for c in dp._nccl.calls) == total
    assert len(dp._nccl.calls) == (total + 3) // 4, "必须按 4 元素一块分块"
    kern = "bf16" if dtype == "bf16" else "f16"
    assert "cast_f32_to_%s" % kern in rt.kernel_calls
    assert "cast_%s_to_f32" % kern in rt.kernel_calls
    assert dp.comm_rank_bytes == total * 2
    assert dp.comm_rank_bytes_fp32 == total * 4


def test_gpu_pipeline_ef_residual_and_second_step(monkeypatch):
    """EF：残差 = g − dequant(quant(g))；下一步把它加回，结果必须等于按残差重算的值。"""
    monkeypatch.delenv("TG_DP_TIMING", raising=False)
    monkeypatch.setattr(D, "_lowp_chunk_elems", lambda total: 3)
    dp, rt = _fake_dp(monkeypatch, "bf16", grad_ef=True)
    v = np.array([1.01171875, -2.3, 0.0078125, 3.14159265, 0.375], np.float32)
    entries = _entries(rt, [v])
    total = int(v.size)
    dp._all_reduce_lowp(entries, total, None)

    resid = dp._ef_dev_buf
    assert resid is not None and resid.np.size == total, "EF 必须分配 fp32 影子"
    own_q = D.dequantize_from_bits(D.quantize_to_bits(v, "bf16"), "bf16")
    np.testing.assert_array_equal(resid.np[:total], v - own_q,
                                  err_msg="EF 残差必须等于 g − dequant(quant(g))")

    # 第二步：同一份梯度（模拟新的 backward），EF 把残差加回去
    g = entries[0][1]
    g._gpu.np[:] = v
    dp._all_reduce_lowp(entries, total, None)
    g_eff = v + (v - own_q)
    q_eff = D.dequantize_from_bits(D.quantize_to_bits(g_eff, "bf16"), "bf16")
    three = q_eff + q_eff * np.float32(2.0)
    want = (D.dequantize_from_bits(D.quantize_to_bits(three, "bf16"), "bf16")
            * np.float32(0.5))
    np.testing.assert_array_equal(g._gpu.np[:total], want)


def test_gpu_pipeline_missing_grad_is_zeroed(monkeypatch):
    """某 rank 缺梯度：仍以 0 参与集合通信（count 不变），最后补全 0 梯度。"""
    monkeypatch.delenv("TG_DP_TIMING", raising=False)
    monkeypatch.setattr(D, "_lowp_chunk_elems", lambda total: 1000)
    dp, rt = _fake_dp(monkeypatch, "bf16")
    entries = _entries(rt, [np.array([1.1, -2.3, 0.0078125], np.float32)])
    missing_p = _FakeParam(4)
    missing_p.grad = None
    entries.append((missing_p, None, 4))
    total = sum(e[2] for e in entries)
    saved = {}

    def _set_grad(p, val, orig):
        saved["missing"] = np.array(val, np.float32)
        p.grad = val

    monkeypatch.setattr(D, "_set_grad", _set_grad)
    dp._all_reduce_lowp(entries, total, None)
    assert sum(c[0] for c in dp._nccl.calls) == total, (
        "缺梯度的位置也必须占 count（否则各 rank 的集合通信长度不同）")
    np.testing.assert_array_equal(saved["missing"], np.zeros((4,), np.float32))
