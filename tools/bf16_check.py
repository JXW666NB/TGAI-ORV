# -*- coding: utf-8 -*-
"""GPU 上验证 bf16 AMP：要么真的跑 bf16（结果接近 fp32），要么**优雅回退**（结果与 fp32 完全相同）。

判据：绝不出现"既不是 bf16 精度、也不是 fp32 结果"的第三种情况（那就是静默算错）。

运行（需独占 GPU）：python bf16_check.py
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log


def rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(1e-9, float(np.abs(b).max())))


def main():
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    rng = np.random.default_rng(0)
    a_np = rng.normal(0, 0.1, (2048, 2048)).astype(np.float32)
    b_np = rng.normal(0, 0.1, (2048, 6144)).astype(np.float32)

    T.amp_off()
    T.amp_off(dtype="bf16")
    a, b = T.tensor(a_np), T.tensor(b_np)
    ref = np.asarray((a @ b).to_numpy(), np.float32)
    log("fp32 基线完成")

    with T.amp(dtype="bf16"):
        st = T.amp_state()
        log("amp 状态: bf16=%s fp16=%s" % (st.get("bf16"), st.get("fp16")))
        got = np.asarray((T.tensor(a_np) @ T.tensor(b_np)).to_numpy(), np.float32)

    r = rel(got, ref)
    if r == 0.0:
        log("结果与 fp32 **完全一致** ⇒ 运行期已回退 fp32（V100 无 bf16 硬件，符合预期）")
        verdict = "FALLBACK"
    elif r <= 5e-2:
        log("结果与 fp32 相对差 %.2e ⇒ bf16 真的生效了（bf16 只有 8 位尾数，1e-2 量级正常）" % r)
        verdict = "BF16"
    else:
        log("结果与 fp32 相对差 %.2e ⇒ **异常**：既不是 bf16 精度也不是 fp32 结果" % r)
        verdict = "BAD"

    log("结论：%s（%s）" % (verdict, "PASS" if verdict in ("FALLBACK", "BF16") else "FAIL"))
    return 0 if verdict in ("FALLBACK", "BF16") else 1


if __name__ == "__main__":
    sys.exit(main())
