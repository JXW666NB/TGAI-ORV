# -*- coding: utf-8 -*-
"""跑 `dist_gpu_probe.py` 的两个 rank，并**对拍**单进程等效结果。

判据（见 probe 文档）：
    广播：两个 rank 的 p/q 都等于 rank0 的初值（p=1, q=7）
    p 梯度：rank0=1、rank1=3 ⇒ 平均 2.0
    q 梯度：rank0=5、rank1 缺梯度 ⇒ 平均 2.5   ← 缺梯度按 0 补齐

用法：`python tools/dist_gpu_check.py [--wait 30]`
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=int, default=180, help="每个 rank 的超时秒数")
    a = ap.parse_args()

    job = "gpu_%d" % (int(time.time()) % 100000)
    outdir = tempfile.mkdtemp(prefix="dp_gpu_")
    procs = []
    for rank in (0, 1):
        procs.append(subprocess.Popen(
            [sys.executable, "-X", "utf8",
             os.path.join(HERE, "dist_gpu_probe.py"), str(rank), "2", job, outdir],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", cwd=REPO))
    outs = []
    for p in procs:
        try:
            out, _ = p.communicate(timeout=a.wait)
        except subprocess.TimeoutExpired:
            p.kill()
            out = "<超时被杀>"
        outs.append((p.returncode, out))

    failed = False
    for i, (rc, out) in enumerate(outs):
        if rc != 0:
            failed = True
            print("rank%d rc=%s\n%s" % (i, rc, (out or "")[-1500:]))

    res = []
    if not failed:
        for rank in (0, 1):
            p = os.path.join(outdir, "rank%d.json" % rank)
            if not os.path.isfile(p):
                print("缺 rank%d 结果文件" % rank)
                failed = True
            else:
                with open(p, encoding="utf-8") as f:
                    res.append(json.load(f))

    ok = not failed and len(res) == 2
    if ok:
        for r in res:
            tag = "rank%d" % r["rank"]
            if r["p_device"] != "cuda":
                print("  ✗ %s 参数不在 cuda 上（device=%s）" % (tag, r["p_device"]))
                ok = False
            if r["p_after"] != [1.0] * 4:
                print("  ✗ %s 广播后 p=%s（应为 1.0，说明 CUDA 广播没生效）"
                      % (tag, r["p_after"][:2]))
                ok = False
            if r["q_after"] != [7.0] * 4:
                print("  ✗ %s 广播后 q=%s（应为 7.0）" % (tag, r["q_after"][:2]))
                ok = False
            if r["p_grad"] != [2.0] * 4:
                print("  ✗ %s p 梯度=%s（应为 2.0 = (1+3)/2）" % (tag, r["p_grad"][:2]))
                ok = False
            if r["q_grad"] != [2.5] * 4:
                print("  ✗ %s q 梯度=%s（应为 2.5 = (5+0)/2，缺梯度按 0 补齐）"
                      % (tag, r["q_grad"][:2]))
                ok = False
            if not r.get("q_grad_is_tensor"):
                print("  ✗ %s 缺梯度一侧的 q.grad 不是张量（CUDA 优化器会报需要 CUDA Tensor）"
                      % tag)
                ok = False
            if r.get("q_grad_device") != "cuda":
                print("  ✗ %s q.grad 落在 %s（应为 cuda）"
                      % (tag, r.get("q_grad_device")))
                ok = False
        if ok:
            r0 = res[0]
            print("GPU DP 对拍 PASS")
            print("  广播后 p=%s q=%s（两个 rank 一致）"
                  % (r0["p_after"][:2], r0["q_after"][:2]))
            print("  平均后 p_grad=%s q_grad=%s（q 在 rank1 无梯度）"
                  % (r0["p_grad"][:2], r0["q_grad"][:2]))
    if not ok:
        print("GPU DP 对拍 **FAIL**")

    shutil.rmtree(outdir, ignore_errors=True)
    shutil.rmtree(os.path.join(tempfile.gettempdir(), "tgai_dp_%s" % job),
                  ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
