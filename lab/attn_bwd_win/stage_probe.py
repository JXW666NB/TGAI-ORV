# -*- coding: utf-8 -*-
"""阶段隔离实验：把 attn_bwd_flash_w 在各阶段后早退，逐档计时。

原理：ORV 支持 TGAI_ATTN_NOP_CU 覆盖内核源码路径（cuda.py:2558），
所以只要生成一份「在第 N 阶段后 return」的变体，用环境变量指过去，
就能让引擎编译并运行该变体 —— 不改包、不改仓库内核。

用法：
    python stage_probe.py 1 2 3 4 5 6 7     # 依次跑各阶段
    python stage_probe.py --child N          # 内部用

阶段锚点（按内核真实代码顺序）：
  1  进入内核后立即 return（只量启动开销）
  2  load_KT(V...)          之后   -> 加上 K/V 载入
  3  load_win(dO...)        之后   -> 加上 Q/dO 载入
  4  flash_qk_dp_win(...)   之后   -> 加上 QKᵀ + dP（主算力）
  5  smDS[...] = ds         之后   -> 加上 dS 写回
  6  dQ 的 atomicAdd        之后   -> 加上 dQ 累加
  7  完整内核
"""
import io
import os
import re
import subprocess
import sys
import tempfile

ANCHORS = {
    2: "load_KT(V, smV, k0, h, nh, dh, tid, seq);",
    3: "load_win(dO, smDO, t0, h, nh, dh, tid, seq, d0, DCHUNK, QSTR);",
    4: "flash_qk_dp_win(smQ, smDO, smK, smV, row0, lane, d0, sx, dp);",
    5: "smDS[(row0 + r) * BC + lane] = ds;",
}

SRC = os.environ.get("ORV_KERNEL_SRC") or "/nonexistent"


def make_variant(stage, dst):
    src = io.open(SRC, encoding="utf-8").read()
    i = src.index("attn_bwd_flash_w")
    head, body = src[:i], src[i:]
    if stage == 1:
        body = re.sub(r"(\{)\s*\n", r"\1\n        if (1) return;\n", body, count=1)
    elif stage == 6:
        m = re.search(r"atomicAdd\([^\n]*", body)
        if not m:
            raise SystemExit("找不到 atomicAdd")
        a = m.group(0)
        body = body.replace(a, a + "\n        if (1) return;", 1)
    else:
        a = ANCHORS[stage]
        if a not in body:
            raise SystemExit("锚点不存在 stage=%d" % stage)
        ind = body[:body.index(a)].rsplit("\n", 1)[-1]
        body = body.replace(a, a + "\n" + ind + "if (1) return;", 1)
    io.open(dst, "w", encoding="utf-8", newline="\n").write(head + body)


CHILD = r'''
import time, numpy as np
import tgai_go.train as T
F = T.functional
T.set_default_device("cuda"); rt = T._cuda()
B, S, D, NH, HD = 4, 2048, 1152, 18, 64
BH = B * NH
rng = np.random.default_rng(0)
rnd = lambda *sh: T.tensor(rng.normal(0, 0.02, sh).astype(np.float32), requires_grad=True)
q, k, v = rnd(BH, S, HD), rnd(BH, S, HD), rnd(BH, S, HD)
ones = T.tensor(np.ones((BH, S, HD), np.float32))
def run():
    q.grad = k.grad = v.grad = None
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, lowmem=True)
    y.backward(ones)
run(); rt.sync()
n = 5
t0 = time.time()
for _ in range(n):
    run(); rt.sync()
print("MS %.2f" % ((time.time() - t0) / n * 1000.0), flush=True)
'''


def main():
    if sys.argv[1:2] == ["--child"]:
        exec(compile(CHILD, "<child>", "exec"))
        return 0
    stages = [int(x) for x in sys.argv[1:]] or list(range(1, 8))
    tmp = tempfile.mkdtemp(prefix="stg_")
    prev = None
    print("stage  ms     delta", flush=True)
    for st in stages:
        dst = os.path.join(tmp, "stage%d.cu" % st)
        make_variant(st, dst)
        env = dict(os.environ)
        env["TGAI_ATTN_NOP_CU"] = dst
        p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--child"],
                           env=env, capture_output=True, text=True)
        out = (p.stdout or "").strip().splitlines()
        line = [l for l in out if l.startswith("MS ")]
        if not line:
            print("%-5d FAILED %s" % (st, (p.stderr or "").strip().splitlines()[-1:] or "?"), flush=True)
            prev = None
            continue
        ms = float(line[0].split()[1])
        d = "-" if prev is None else "%.2f" % (ms - prev)
        print("%-5d %6.2f %s" % (st, ms, d), flush=True)
        prev = ms
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
