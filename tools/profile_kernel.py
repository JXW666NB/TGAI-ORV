# -*- coding: utf-8 -*-
"""内核级剖析：用 CUDA event 量每个 kernel 的真实 GPU 时间（相对整步 wall）。

为什么需要它：per-op 剖析只能看到前向；审计的毫秒数按 95 GB/s 估的（实测 500+），
都被证伪过。这个工具给出**真实账**：GPU 到底忙了多久、CPU 空转多少。

实测结论（2026-11-02，V100，B=4/S=512，Qwen2.5-0.5B 形状）：
    wall = 744 ms，kernel 数 1330，GPU 忙 **111.6 ms**  =>  **85% 时间是 CPU 发射开销**
    最大单项：adamw_update 290 次 18.8 ms（16.8%）、sum_axis8 72 次 14.5 ms（13%）
=> 下一步不是继续优化 kernel，而是减少发射次数（CUDA Graph）或降 CPU 开销。

用法：
    python tools/profile_kernel.py --shape 4x512 --checkpoint <safetensors 目录>
"""
import argparse
import ctypes
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="safetensors 目录（含 model.safetensors）")
    ap.add_argument("--shape", default="4x512", help="BxS，例如 4x512")
    ap.add_argument("--top", type=int, default=14)
    a = ap.parse_args(argv)
    b, s = (int(x) for x in a.shape.lower().split("x"))

    import numpy as np
    import tgai_go.train as T
    from tgai_go.train import checkpoints as C
    from tgai_go.train.cuda import CudaRuntime

    T.set_default_device("cuda")
    from tgai_go.train import qwen as Q

    orig = CudaRuntime.launch
    events = []
    on = {"v": False}

    def prof_launch(self, name, grid, block, args, shared=0, module=None):
        if not on["v"]:
            return orig(self, name, grid, block, args, shared, module)
        e0, e1 = ctypes.c_void_p(), ctypes.c_void_p()
        ok = self.cu.cuEventCreate(ctypes.byref(e0), 0) == 0
        if ok:
            self.cu.cuEventRecord(e0, self._stream)
        out = orig(self, name, grid, block, args, shared, module)
        if ok and self.cu.cuEventCreate(ctypes.byref(e1), 0) == 0:
            self.cu.cuEventRecord(e1, self._stream)
            events.append((name, e0, e1))
        elif ok:
            self.cu.cuEventDestroy(e0)
        return out

    CudaRuntime.launch = prof_launch

    ck = C.load_safetensors_dir(a.checkpoint)
    cfg = dict(Q.cfg_from_checkpoint(ck))
    model = Q.Qwen25(cfg, dict(ck.weights))
    del ck
    opt = T.optim.AdamW(model.params(), lr=5e-6, betas=(0.9, 0.95), eps=1e-8,
                        weight_decay=0.0)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 1000, (b, s)).astype(np.int64)
    tgt = rng.integers(0, 1000, (b * s,)).astype(np.int64)
    rt = T._cuda()

    def step():
        lg = model.forward(ids)
        ls = T.functional.cross_entropy(lg.reshape(b * s, cfg["V"]), tgt)
        ls.backward()
        opt.step()
        opt.zero_grad()

    step()
    events.clear()
    on["v"] = True
    t0 = time.perf_counter()
    step()
    on["v"] = False
    rt.sync()
    wall = time.perf_counter() - t0

    stat = {}
    for name, e0, e1 in events:
        ms = ctypes.c_float()
        if rt.cu.cuEventElapsedTime(ctypes.byref(ms), e0, e1) == 0:
            ent = stat.setdefault(name, [0, 0.0])
            ent[0] += 1
            ent[1] += float(ms.value)
        rt.cu.cuEventDestroy(e0)
        rt.cu.cuEventDestroy(e1)

    gpu = sum(v[1] for v in stat.values())
    print("wall=%.1f ms  kernels=%d  gpu_sum=%.1f ms  gpu_util=%.1f%%"
          % (wall * 1e3, len(events), gpu, gpu / (wall * 1e3) * 100))
    for name, (cnt, ms) in sorted(stat.items(), key=lambda kv: -kv[1][1])[:a.top]:
        print("  %-26s %5d 次 %8.2f ms %5.1f%%" % (name, cnt, ms, ms / gpu * 100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
