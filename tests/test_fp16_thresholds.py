# -*- coding: utf-8 -*-
"""fp16 GEMM 阈值默认值（0.9.19）：K 上限必须放开到大数，否则大形状退回 fp32。

为什么要这条测试：旧默认 `_FP16_MAX_K = 1280` 把 tensor core **最有用**的大 K 形状全挡回 fp32。
实测（V100，fp16 输入 + fp32 累加）：
    2048x2048x6144 : 3.86 ms → 0.67 ms
    2048x2048x8192 : 5.30 ms → 0.94 ms
    2048x4096x11008: 13.89 ms → 2.13 ms
    2048x2048x32000: 19.24 ms → 3.15 ms
且相对 fp32 的误差 2.78e-04~2.88e-04，与 torch AMP **同值**。

注意：`cuda.py` 在 import 时被 exec 进一个私有命名空间（`tgai_go.train.cuda` 这个名字被
`_CudaAPI` 实例遮蔽，`sys.modules` 里也没有），所以这里**读源码**来断言默认值，而不是 import。
"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUDA_PY = os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py")


def _default(name):
    src = io.open(CUDA_PY, encoding="utf-8").read()
    m = re.search(r'(?m)^%s\s*=\s*([^\n#]+)' % re.escape(name), src)
    if not m:
        raise AssertionError("cuda.py 里找不到 %s 的默认值定义" % name)
    return m.group(1).strip()


def test_fp16_k_threshold_default_is_large():
    if os.environ.get("TG_FP16_MAX_K"):
        import pytest
        pytest.skip("环境变量显式覆盖了阈值")
    expr = _default("_FP16_MAX_K")
    assert "1 << 30" in expr or "1000000000" in expr, (
        "fp16 GEMM 的 K 默认上限太小（%r）：大形状会退回 fp32，tensor core 用不上。"
        "修法：默认保持 1 << 30（可用 TG_FP16_MAX_K 覆盖）" % (expr,))


def test_fp16_min_output_keeps_small_gemms_in_fp32():
    expr = _default("_FP16_MIN_OUTPUT")
    assert int(expr) >= (1 << 16), (
        "小 GEMM 不应该转 fp16（cast 比 GEMM 还贵），_FP16_MIN_OUTPUT 应保持 262144 量级")
