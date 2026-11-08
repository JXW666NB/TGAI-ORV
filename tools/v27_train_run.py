# -*- coding: utf-8 -*-
"""V2.7 全参训练入口（V100 单卡，不借主机内存）。

用法（在 V100 上）：
    python v27_train_run.py --steps 200 --seq 512 --lr 1e-5

它会：
1. **隔离 GPU**：临时改名 launcher、停 8 个计划任务、杀 ka_loop 进程、轮询到 GPU 空闲
2. 流式装载 28 层（11.3 GB）+ LM head，建 fp16 动量优化器（22.6 GB）
3. 逐层反向训练 `--steps` 步（每层算完立即 step 并释放梯度）
4. `finally` 里**恢复产线**（改回 launcher、启用任务、起回 11450/11451）

为什么要"改名 launcher"：这台机器上的计划任务/保活会在我停服务后 30 秒内把推理拉回来，
占走 27 GB，导致训练中途 OOM。改名是从根上断掉复活路径，跑完自动改回。
"""
import argparse
import os
import sys

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")
import subprocess
import time

import numpy as np
import torch

TASKS = ("TGAI-GO-11450", "TGAI-GO-11451", "TGAI-GO-11440", "TGAI-GO-11441",
         "TGAI-KEEPALIVE", "TGAI-INFER-WATCH", "TGAI-PROD-INFER", "TGAI-ENGINE-BOOT")
LAUNCHERS = (r"D:\tgai_orv\launch11450.py", r"D:\tgai_orv\launch11451.py")
PIDFILE = r"D:\TGAI\05_logs\ka_loop.pid"
PT = r"D:\TGAI\models\V2.7\tgai_v27_v5_merged.pt"
LOG = r"D:\tgai_orv\v27_train_run.log"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("%s %s\n" % (time.strftime("%H:%M:%S"), msg))


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True,
                          encoding="utf-8", errors="ignore").stdout


def vram():
    return int(sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader").split()[0])


def gpu_procs():
    out = sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader").strip()
    return len([x for x in out.splitlines() if x.strip()])


def isolate():
    for t in TASKS:
        sh('schtasks /change /tn "%s" /disable' % t)
        sh('schtasks /end /tn "%s"' % t)
    for f in LAUNCHERS:
        if os.path.isfile(f):
            os.rename(f, f + ".off")
    if os.path.isfile(PIDFILE):
        pid = open(PIDFILE).read().strip()
        if pid:
            sh("taskkill /F /PID %s" % pid)
    sh("wmic process where \"name='python.exe' and commandline like '%ka_loop%'\" delete")
    for port in (11450, 11451):
        out = sh("netstat -ano | findstr :%d | findstr LISTENING" % port).strip()
        if out:
            sh("taskkill /F /PID %s" % out.split()[-1])
    for _ in range(20):
        time.sleep(5)
        isolate_soft()
        if vram() < 100 and gpu_procs() == 0:
            time.sleep(5)
            if vram() < 100 and gpu_procs() == 0:
                return True
    return False


def isolate_soft():
    """再压一次（任务可能被时间触发器重新拉起）。"""
    for t in TASKS:
        sh('schtasks /end /tn "%s"' % t)


def restore():
    for f in LAUNCHERS:
        if os.path.isfile(f + ".off"):
            os.rename(f + ".off", f)
    for t in TASKS:
        sh('schtasks /change /tn "%s" /enable' % t)
    for t in ("TGAI-GO-11440", "TGAI-GO-11441"):
        sh('schtasks /Run /TN "%s"' % t)
    for f in LAUNCHERS:
        if os.path.isfile(f):
            subprocess.Popen([r"C:\Python311\python.exe", "-u", f],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--no-restore", action="store_true", help="不恢复产线（调试用，危险）")
    args = ap.parse_args()

    open(LOG, "w").close()
    log("=== v27 全参训练 steps=%d seq=%d lr=%g ===" % (args.steps, args.seq, args.lr))
    if not isolate():
        log("GPU 无法隔离（%d MiB, procs=%d），放弃" % (vram(), gpu_procs()))
        restore()
        raise SystemExit(3)
    log("GPU 已隔离：%d MiB procs=%d" % (vram(), gpu_procs()))

    try:
        import tgai_go.train as T
        T.set_default_device("cuda")
        from tgai_go.train.opt_fp16mv import Fp16MomAdamW
        from tgai_go.train.v27_train import V27Trainer

        ck = torch.load(PT, map_location="cpu", mmap=True, weights_only=False)
        mc, sd = ck["model_config"], ck["model_state_dict"]
        cfg = dict(d_model=int(mc["d_model"]), n_heads=int(mc["n_heads"]),
                   d_ff=int(mc["d_ff"]), n_experts=int(mc["n_experts"]),
                   n_activated=int(mc["n_activated"]), n_layers=int(mc["n_layers"]),
                   rope_theta=float(mc.get("rope_theta", 1e4)))
        NL = cfg["n_layers"]

        t0 = time.time()
        tr = V27Trainer(cfg, n_layers=NL)
        tr.load_all(sd)
        tr.attach_head(sd)
        log("装载 28 层 + head：%d MiB，%.1fs" % (vram(), time.time() - t0))

        opts = [Fp16MomAdamW(tr.params_of(i), lr=args.lr) for i in range(NL)]
        head_opt = Fp16MomAdamW(tr.head_params(), lr=args.lr)
        log("优化器就绪（fp16 动量）：%d MiB（%.1f GB）" % (vram(), vram() / 1024))

        D = cfg["d_model"]
        DH = D // cfg["n_heads"]
        TK = args.seq
        j = np.arange(0, DH, 2, dtype=np.float32) / DH
        ang = np.repeat(np.arange(TK, dtype=np.float32)[:, None]
                        * (cfg["rope_theta"] ** (-j))[None, :], 2, axis=1)
        cos = T.tensor(np.cos(ang).astype(np.float32))
        sin = T.tensor(np.sin(ang).astype(np.float32))
        x0 = np.random.default_rng(0).normal(0, .1, (TK, D)).astype(np.float32)
        V = int(sd["token_embedding.weight"].shape[0])
        targets = np.random.default_rng(2).integers(0, V, TK)

        losses = []
        t1 = time.time()
        for step in range(args.steps):
            xs = tr.forward_all(x0, cos, sin)
            loss, g = tr.loss_and_head_backward(xs[-1], targets, cos, sin)
            lv = float(np.asarray(loss.to_numpy() if hasattr(loss, "to_numpy") else loss))
            losses.append(lv)
            head_opt.step()
            head_opt.zero_grad()
            for p in tr.head_params():
                p.grad = None
            tr.backward_and_step(xs, g, cos, sin, opts)
            if step % args.log_every == 0 or step == args.steps - 1:
                log("step %4d  loss=%.4f  显存 %d MiB  %.2fs"
                    % (step, lv, vram(), time.time() - t1))
        el = time.time() - t1
        log("完成：loss %.4f → %.4f（降 %.1f%%），%d 步 %.1fs，均值 %.2fs/步"
            % (losses[0], losses[-1], (1 - losses[-1] / losses[0]) * 100,
               args.steps, el, el / args.steps))
        log("峰值显存 %d MiB（%.1f GB / 32 GB）" % (vram(), vram() / 1024))
    finally:
        if not args.no_restore:
            restore()
            for _ in range(24):
                time.sleep(5)
                if sh("netstat -ano | findstr :11450 | findstr LISTENING").strip():
                    break
            log("产线已恢复：显存 %d MiB" % vram())


if __name__ == "__main__":
    main()
