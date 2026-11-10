# -*- coding: utf-8 -*-
"""fp16 GEMM 的数值与速度核对：与 fp32 的相对误差、与 torch AMP 的一致性。

判据：fp16 输入（fp32 累加）相对 fp32 的误差应在 1e-3 量级（fp16 有 ~11 位有效位），
不是"逐位一致"，所以这里给的是**误差量级**而不是 0.00e+00。

运行（需独占 GPU）：python fp16_numerics.py
"""
import os
import sys

os.environ["TG_FP16"] = "1"
sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

_spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

SHAPES = [(512, 1024, 4096), (2048, 2048, 6144), (2048, 4096, 11008)]


def rel(a, b):
    import numpy as np
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / max(1e-9, float(np.abs(b).max())))


def main():
    import numpy as np
    import torch
    import tgai_go.train as T
    T.set_default_device("cuda")
    rt = T._cuda()
    log("fp16 状态: %r" % (rt._fp16_state(),))
    ok = True
    for M, K, N in SHAPES:
        rng = np.random.default_rng(0)
        a_np = rng.normal(0, 0.1, (M, K)).astype(np.float32)
        b_np = rng.normal(0, 0.1, (K, N)).astype(np.float32)

        # fp32 参照（明确关掉 fp16）
        st = rt._fp16_state()
        rt._fp16_state()["on"] = False
        a = T.tensor(a_np)
        b = T.tensor(b_np)
        y32 = np.asarray((a @ b).to_numpy(), np.float32)
        rt._fp16_state()["on"] = True

        # fp16 路径
        y16 = np.asarray((T.tensor(a_np) @ T.tensor(b_np)).to_numpy(), np.float32)

        # torch AMP 参照
        ta = torch.from_numpy(a_np)
        tb = torch.from_numpy(b_np)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            ty = (ta @ tb).float().cpu().numpy()

        r_self = rel(y16, y32)
        r_pyt = rel(y16, ty)
        log("%5dx%5dx%-6d | fp16-vs-fp32 %.2e | fp16-vs-torchAMP %.2e" % (M, K, N, r_self, r_pyt))
        ok = ok and r_self <= 5e-3 and r_pyt <= 5e-3
        del a, b
        rt.empty_cache()
    log("总结论：%s（fp16 输入 + fp32 累加，误差应在 1e-3 量级）" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
