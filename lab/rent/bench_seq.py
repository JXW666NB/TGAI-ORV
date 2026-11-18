# -*- coding: utf-8 -*-
"""V100 上跑 seq 扫描 + 分块阈值 A/B + 模型大小影响。

只做一件事：调起两个框架的训练脚本，解析它们打印的 RESULT JSON，汇总成表。
用 Python 而不是 PowerShell，避免 PS 5.1 对中文/引号的解析坑。
"""
import json
import os
import re
import subprocess
import sys

ROOT = r"D:\orv_bench"
ORV = r"D:\orv_pip_0962\Scripts\python.exe"
PYT = r"C:\Python311\python.exe"
STEPS = 4


def run(cmd, env_extra=None):
    env = dict(os.environ)
    env.pop("TG_CSA_ATTN_CQ", None)
    if env_extra:
        env.update(env_extra)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, cwd=ROOT)
    return (p.stdout or "") + (p.stderr or "")


def parse_orv(out):
    m = re.search(r'^RESULT (\{.*\})$', out, re.M)
    if not m:
        why = "OOM" if ("显存不足" in out or "out of memory" in out.lower()) else "FAIL"
        return None, why
    d = json.loads(m.group(1))
    return d, None


def parse_pyt(out):
    m = re.search(r'^RESULT (\{.*\})$', out, re.M)
    if not m:
        why = "OOM" if "out of memory" in out.lower() else "FAIL"
        return None, why
    return json.loads(m.group(1)), None


def orv(L, DFF, E, TK, SEQ, cq=None):
    cmd = [ORV, "-m", "tgai_go.train.v3_pretrain",
           "--data", os.path.join(ROOT, "data2048"), "--vocab", "96000",
           "--seq", str(SEQ), "--d", "1024", "--layers", str(L), "--E", str(E),
           "--n_q", "16", "--d_ff", str(DFF), "--top_k", str(TK), "--csa", "4",
           "--opt", "int8gpu", "--steps", str(STEPS), "--log-every", "1",
           "--no-final-save"]
    env = {"TG_CSA_ATTN_CQ": str(cq)} if cq is not None else None
    d, why = parse_orv(run(cmd, env))
    if d is None:
        return "ORV  %-22s cq=%-7s FAIL: %s" % (_tag(L, SEQ), _cq(cq), why)
    return ("ORV  %-22s cq=%-7s %6.3f s  %8.1f tok/s  %6.2f GiB"
            % (_tag(L, SEQ), _cq(cq), d["step_s_steady"], d["tok_s_steady"],
               d["pool_peak_live_gib"]))


def pyt(L, DFF, E, TK, SEQ):
    cmd = [PYT, os.path.join(ROOT, "v3_pyt_train.py"), "--d", "1024",
           "--layers", str(L), "--E", str(E), "--vocab", "96000",
           "--d_ff", str(DFF), "--n_heads", "16", "--top_k", str(TK),
           "--seq", str(SEQ), "--steps", str(STEPS)]
    d, why = parse_pyt(run(cmd))
    if d is None:
        return "PyT  %-22s              FAIL: %s" % (_tag(L, SEQ), why)
    return ("PyT  %-22s              %6.3f s  %8.1f tok/s  %6.2f GiB"
            % (_tag(L, SEQ), d["step_s"], SEQ / d["step_s"], d["peak_mib"] / 1024.0))


def _tag(L, SEQ):
    return "moe%d-seq%d" % (L, SEQ)


def _cq(cq):
    return "default" if cq is None else str(cq)


def main():
    print("===== A. seq 扫描（MoE 12 层，ORV 默认 cq=2048）=====")
    for S in (512, 1024, 2048):
        print(orv(12, 3328, 4, 2, S))
        print(pyt(12, 3328, 4, 2, S))

    print("\n===== B. 分块阈值 A/B（看是不是分块没触发导致的）=====")
    for S in (2048, 1024):
        for CQ in (512, 256):
            print(orv(12, 3328, 4, 2, S, cq=CQ))
        print(pyt(12, 3328, 4, 2, S))

    print("\n===== C. 模型大小的影响（seq 2048）=====")
    for L in (8, 12, 24):
        print(orv(L, 3328, 4, 2, 2048, cq=256))
        print(pyt(L, 3328, 4, 2, 2048))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
