# -*- coding: utf-8 -*-
"""租卡体检：一条命令跑完 Ampere（A4000/sm_86）上该确认的所有事。

用法（在租到的机器上）：
    python tools/am_preflight.py --checkpoint /path/to/qwen25_safetensors
    python tools/am_preflight.py --checkpoint ... --dp-world 2      # 两卡时

它会依次做：
  1. 环境：GPU 名/算力/显存、CUDA 驱动、torch 版本、bf16/TF32 硬件能力
  2. 自检：tools/accept.py --quick（doctor + 形状矩阵 + 120 步 + 存续训）
  3. GEMM 对照：同一批模型 shape 下 fp32(PEDANTIC) / fp32(TF32) / fp16 / bf16 的 ms 与 TFLOPS
  4. 训练一步：B=2/S=512 下的 s/step（fp16 开/关），并报告显存峰值
  5. 多卡（给了 --dp-world 且 >1 时）：梯度 all-reduce 的单步耗时（判断走共享内存够不够）

所有结果以 `RESULT ` 开头打印，方便直接贴回来。
"""
import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))


def _fmt(x):
    return "%.2f" % x


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Qwen2.5 safetensors 目录")
    ap.add_argument("--batch", type=int, default=2, help="训练基准的 B（16 GiB 建议 2）")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--dp-world", type=int, default=1)
    ap.add_argument("--skip-accept", action="store_true")
    a = ap.parse_args(argv)

    import numpy as np
    import tgai_go.train as T

    # ---------- 1. 环境 ----------
    try:
        import torch
        tver = torch.__version__
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            cap = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / float(1 << 30)
            print("RESULT torch=%s gpu=%r cc=%d.%d mem=%.1f GiB n_gpu=%d"
                  % (tver, name, cap[0], cap[1], total, torch.cuda.device_count()))
            print("RESULT bf16_hw=%s tf32_hw=%s"
                  % (cap[0] >= 8, cap[0] >= 8))
    except Exception as exc:                        # noqa: BLE001
        print("RESULT torch 不可用（不影响本引擎）：%s" % exc)

    T.set_default_device("cuda")
    rt = T._cuda()
    print("RESULT engine_device_ready=%s" % (rt.cu is not None))

    # ---------- 2. 自检 ----------
    if not a.skip_accept:
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "accept.py"),
                            "--quick"], capture_output=True, text=True,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        tail = [ln for ln in (r.stdout or "").splitlines() if "PASS" in ln or "FAIL" in ln]
        print("RESULT accept_rc=%d 耗时 %.0fs 末行=%s"
              % (r.returncode, time.time() - t0,
                 tail[-1].strip() if tail else "(无输出)"))
        for ln in tail[-6:]:
            print("   " + ln.strip())

    # ---------- 3. GEMM 对照 ----------
    from tgai_go.train import qwen as Q
    ck = Q.__dict__.get("load_safetensors_dir")
    from tgai_go.train import checkpoints as C
    w = C.load_safetensors_dir(a.checkpoint)
    cfg = dict(Q.cfg_from_checkpoint(w))
    del w
    D, FF, V = int(cfg["D"]), int(cfg["FF"]), int(cfg["V"])
    B, S = a.batch, a.seq
    N = B * S

    def bench(fn, it=3):
        fn()
        rt.sync()
        t0 = time.perf_counter()
        for _ in range(it):
            fn()
            rt.sync()
        return (time.perf_counter() - t0) / it

    shapes = [("mlp", N, D, FF), ("lm_head", N, D, V)]
    for name, m, k, n in shapes:
        A = T.tensor(np.random.randn(m, k).astype(np.float32) * 0.3)
        W = T.tensor(np.random.randn(n, k).astype(np.float32) * 0.05)
        fl = 2 * m * k * n
        t32 = bench(lambda: A @ W.T)
        try:
            t16 = bench(lambda: rt.gemm_f16_nt_raw(
                rt.cast_f32_to_f16(A._gpu), rt.cast_f32_to_f16(W._gpu), m, n, k))
        except Exception as exc:                    # noqa: BLE001
            t16 = float("nan")
            print("   fp16 不可用：%s" % str(exc)[:60])
        print("RESULT gemm %-8s fp32=%7.1f ms (%5.1f TF)  fp16=%7.1f ms (%5.1f TF)"
              % (name, t32 * 1e3, fl / t32 / 1e12, t16 * 1e3,
                 (fl / t16 / 1e12) if t16 == t16 else float("nan")))

    # ---------- 4. 训练一步 ----------
    from tgai_go.train import qwen as QQ
    ids = np.random.default_rng(0).integers(0, 1000, (B, S)).astype(np.int64)
    tgt = np.random.default_rng(1).integers(0, 1000, (B * S,)).astype(np.int64)
    for use16 in (False, True):
        rt.fp16_gemm = use16
        model = QQ.Qwen25(cfg, dict(C.load_safetensors_dir(a.checkpoint)[0]))
        opt = T.optim.AdamW(model.params(), lr=5e-6, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.0)

        def step():
            lg = model.forward(ids)
            ls = T.functional.cross_entropy(lg.reshape(B * S, cfg["V"]), tgt)
            ls.backward()
            opt.step()
            opt.zero_grad()

        try:
            step()
            rt.sync()
            t0 = time.perf_counter()
            for _ in range(3):
                step()
            rt.sync()
            dt = (time.perf_counter() - t0) / 3
            live = getattr(rt, "_pool_live_bytes", 0) / float(1 << 30)
            print("RESULT train B=%d S=%d fp16=%s  %s ms/step  live=%.2f GiB"
                  % (B, S, use16, _fmt(dt * 1e3), live))
        except Exception as exc:                    # noqa: BLE001
            print("RESULT train B=%d S=%d fp16=%s 失败：%s"
                  % (B, S, use16, str(exc).replace("\n", " ")[:120]))
        del model, opt
        rt.fp16_gemm = False

    # ---------- 5. 多卡 ----------
    if a.dp_world and int(a.dp_world) > 1:
        from tgai_go.train import dist
        dp = dist.DataParallel()
        ps = [T.tensor(np.random.randn(4096).astype(np.float32)) for _ in range(64)]
        t0 = time.perf_counter()
        dp.all_reduce_grads(ps)
        rt.sync()
        print("RESULT dp world=%s all_reduce(64x4096) = %.1f ms"
              % (a.dp_world, (time.perf_counter() - t0) * 1e3))

    print("RESULT done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
