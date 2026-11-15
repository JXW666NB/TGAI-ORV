# -*- coding: utf-8 -*-
"""在真机上就地生成「dK/dV 分 plane」的内核变体，并验证它能编译且行为不变。

为什么单独留这个脚本：2026-11-15 的内核改动曾**只在真机副本上验证过**（未进仓库），
如果不落盘，下次要重写一遍。这个脚本把「改哪几处、怎么验证」固化成可执行的形式。

它做的事（**只改副本，不动安装的包**）：
  1. 读安装目录里的 tgai_attn_nop.cu
  2. 用函数名 `attn_bwd_flash_w(` 定位滑窗内核（**不能用裸函数体片段** —— 文件里有
     attn_bwd_flash / attn_bwd_flash_w / attn_bwd_flash_g2 三个结构相似的内核，
     0.9.74 就是锚点匹配错了那个而发布了一个内核编译失败的版本）
  3. 在该函数内：加 `GROUP_DK` 宏、算 plane 指针 dKP/dVP、把 dK/dV 的原子写改到 dKP/dVP
  4. 用 TGAI_ATTN_NOP_CU 指过去，在**子进程**里编译 + 计时
     （子进程保证内核被重新编译，不会命中上一份的缓存）

验收判据（都通过才算这一步成立）：
  * `READY True` —— NVRTC 编译成功（为 False 就是内核坏了）
  * 耗时与旧版一致（P100 上约 89–91 ms）；`GROUP_DK=1` 时偏移为 0，行为必须不变

用法（在有 GPU 的机器上）：
    set SRC=D:\\st\\venv\\Lib\\site-packages\\tgai_go\\train\\kernels\\tgai_attn_nop.cu
    python kernel_plane_patch.py

下一步（尚未做）：启动端要按 `GROUP_DK` 多分配 plane 并做一次归约，才能把 GROUP_DK>1
真正跑起来。规格见 docs/P100_ATTN_BWD_BOTTLENECK.md 第九节。
"""
import io
import os
import re
import subprocess
import sys
import tempfile

SRC = os.environ.get("SRC", "")

CHILD = r'''
import time, numpy as np
import tgai_go.train as T
F = T.functional
T.set_default_device("cuda")
rt = T._cuda()
print("READY", bool(F.sdpa_lowmem_ready()))
B, S, D, NH, HD = 4, 2048, 1152, 18, 64
BH = B * NH
rng = np.random.default_rng(0)
rnd = lambda *sh: T.tensor(rng.normal(0, 0.02, sh).astype(np.float32), requires_grad=True)
q, k, v = rnd(BH, S, HD), rnd(BH, S, HD), rnd(BH, S, HD)
o = T.tensor(np.ones((BH, S, HD), np.float32))
def run():
    q.grad = k.grad = v.grad = None
    F.scaled_dot_product_attention(q, k, v, is_causal=True, lowmem=True).backward(o)
run(); rt.sync()
n = 5
t0 = time.time()
for _ in range(n):
    run(); rt.sync()
print("MS %.2f" % ((time.time() - t0) / n * 1000.0), flush=True)
'''


def make_variant(dst):
    s = io.open(SRC, encoding="utf-8").read()
    i = s.index("attn_bwd_flash_w(")          # ← 必须用函数名定位
    head, body = s[:i], s[i:]

    a = "#define BTB 32"
    assert a in head, "找不到 BTB 定义"
    head = head.replace(a, a + "\n#ifndef GROUP_DK\n#define GROUP_DK 1\n#endif", 1)

    m = re.search(r"if \(t0 >= seq \|\| dh != DH\) return;", body)
    assert m, "找不到滑窗版入口"
    entry = ("\n    const int gpl = (GROUP_DK > 1) ? ((int)blockIdx.y % GROUP_DK) : 0;\n"
             "    const long gpsz = (long)nh * seq * dh;\n"
             "    float* const dKP = dK + (long)gpl * gpsz;\n"
             "    float* const dVP = dV + (long)gpl * gpsz;")
    body = body[:m.end()] + entry + body[m.end():]

    o1 = "float* dkrow = dK + qkv_at(tok, h, nh, dh) + d0 + dl;"
    o2 = "float* dvrow = dV + qkv_at(tok, h, nh, dh) + d0 + dl;"
    assert o1 in body and o2 in body, "找不到 dK/dV 原子行"
    body = body.replace(o1, o1.replace("dK", "dKP"), 1).replace(o2, o2.replace("dV", "dVP"), 1)

    io.open(dst, "w", encoding="utf-8", newline="\n").write(head + body)


def main():
    if sys.argv[1:2] == ["--child"]:
        exec(compile(CHILD, "<child>", "exec"))
        return 0
    if not SRC or not os.path.isfile(SRC):
        raise SystemExit("请先设 SRC=<安装目录里的 tgai_attn_nop.cu>")
    tmp = tempfile.mkdtemp(prefix="plane_")
    dst = os.path.join(tmp, "v.cu")
    make_variant(dst)
    print("变体已生成:", dst, flush=True)
    env = dict(os.environ)
    env["TGAI_ATTN_NOP_CU"] = dst
    p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--child"],
                       env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (p.stdout or "").strip()
    if out:
        print(out, flush=True)
    else:
        err = (p.stderr or "").strip().splitlines()
        print("FAIL", err[-1][:200] if err else "?", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
