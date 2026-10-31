# -*- coding: utf-8 -*-
"""显存/步速剖析：给 AMP 的 S1 提供 **fp32 基线**（bf16 的对照基准）。

为什么现在就要写：AMP 的验收判据是「峰值显存下降 ≥25%」——
**没有 fp32 基线这个判据无从谈起**。基线必须用同一份代码、同一配置测，
不能拿历史数字凑（历史那次 batch=8/seq=256 是抢卡环境，步速不可比）。

做法：起一个短训练（默认 30 步），用 `nvidia-smi` 每 0.2 s 采一次显存，
取峰值；同时记录稳态 s/step。**只采显存与时间，不改训练逻辑。**

用法（A10，GPU 空闲时）：
    python tools/mem_profile.py --safetensors /mnt/workspace/qwen25 \
        --data /mnt/workspace/sft_big.jsonl --tokenizer .../tokenizer.json \
        --seq-len 256 --batch 4 --steps 30 --out /tmp/mem_fp32.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MemSampler:
    """后台线程采样显存（MiB）。用 nvidia-smi，不做任何 CUDA 调用，避免干扰被测进程。"""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._t = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]
                self.samples.append(int(out))
            except Exception:                      # noqa: BLE001
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)
        return False

    @property
    def peak(self):
        return max(self.samples) if self.samples else None

    @property
    def idle(self):
        return min(self.samples) if self.samples else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="fp32")
    a = ap.parse_args(argv)

    idle_before = None
    try:
        idle_before = int(subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0])
    except Exception:                              # noqa: BLE001
        pass

    cmd = [sys.executable, "-X", "utf8", os.path.join(REPO, "examples", "qwen25_sft.py"),
           "--safetensors", a.safetensors, "--data", a.data, "--tokenizer", a.tokenizer,
           "--seq-len", str(a.seq_len), "--batch", str(a.batch), "--steps", str(a.steps),
           "--limit", str(a.limit), "--lr", repr(a.lr), "--clip", "1.0", "--warmup", "5",
           "--device", a.device]
    print("$ " + " ".join(cmd))
    print("（开始前显存占用 %s MiB；采样间隔 0.2 s）" % idle_before)

    t0 = time.time()
    with MemSampler() as ms:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    wall = time.time() - t0
    text = (p.stdout or "") + (p.stderr or "")

    steps = [float(x) for x in re.findall(r"([0-9.]+)\s*s/step", text)]
    rep = {"label": a.label, "rc": p.returncode, "wall_s": round(wall, 1),
           "peak_mib": ms.peak, "idle_mib": ms.idle, "idle_before_mib": idle_before,
           "samples": len(ms.samples),
           "sec_per_step_last": steps[-1] if steps else None,
           "sec_per_step_median": sorted(steps)[len(steps) // 2] if steps else None,
           "seq_len": a.seq_len, "batch": a.batch, "steps": a.steps}

    # 扣掉运行前的占用 = 这次训练自己用到的峰值
    if rep["peak_mib"] and idle_before is not None:
        rep["peak_delta_mib"] = rep["peak_mib"] - idle_before

    print("")
    print("峰值显存 %s MiB，基线 %s MiB ⇒ 本次任务约占 %s MiB"
          % (rep["peak_mib"], idle_before, rep.get("peak_delta_mib")))
    print("步速：中位数 %s s/step，最后一步 %s s/step（wall %.0f s）"
          % (rep["sec_per_step_median"], rep["sec_per_step_last"], wall))
    if p.returncode != 0:
        tail = "\n".join([l for l in text.strip().split("\n") if l.strip()][-6:])
        print("运行失败（rc=%d）：\n%s" % (p.returncode, tail))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print("已写入 %s" % a.out)
    return 0 if p.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
