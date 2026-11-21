# -*- coding: utf-8 -*-
"""AMP round 2: cast attribution, the fused cast, and TF32 (ASCII only)."""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUDA_PY = os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py")
OVERVIEW = os.path.join(ROOT, "python", "tgai_go", "train", "v3_pretrain.py")


def body(name, span=2600):
    src = io.open(CUDA_PY, encoding="utf-8").read()
    i = src.index("def %s" % name)
    seg = src[i:i + span]
    a = seg.find('"""')
    if a >= 0:
        b = seg.find('"""', a + 3)
        if b > a:
            seg = seg[:a] + seg[b + 3:]
    return seg


def test_cast2_kernel_is_registered_in_the_same_source():
    src = io.open(CUDA_PY, encoding="utf-8").read()
    assert "cast2_f32_to_f16" in src
    # 必须复用同一个位转换函数，否则就不是逐位等价
    i = src.index("cast2_f32_to_f16")
    seg = src[i:i + 600]
    assert "tg_f32_to_f16_bits" in seg, "融合内核必须复用 tg_f32_to_f16_bits"


def test_tf32_mode_is_reachable_and_idempotent():
    src = io.open(CUDA_PY, encoding="utf-8").read()
    i = src.index("def tf32_enable")
    seg = src[i:i + 2600]
    assert "ns_math_mode_begin" in seg and "ns_math_mode_end" in seg, (
        "tf32_enable 必须复用已有的 PEDANTIC<->TF32 切换，不要另写一套 cublasSetMathMode")
    assert "_tf32" in seg, "tf32_enable 必须是幂等的（要记住当前状态）"


def test_amp_tf32_is_an_allowed_cli_value():
    src = io.open(OVERVIEW, encoding="utf-8").read()
    m = re.search(r'"--amp",\s*default="([^"]+)",\s*choices=\[([^\]]*)\]', src)
    assert m, "v3_pretrain 的 --amp 定义变了，这个测试要跟着改"
    default, choices = m.group(1), m.group(2)
    assert '"tf32"' in choices, "--amp 必须接受 tf32"
    assert default == "off", "--amp 的默认值必须保持 off"
