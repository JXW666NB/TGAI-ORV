# -*- coding: utf-8 -*-
"""把实验场验证通过的"窗口版反向内核"搬进生产，作为**并存的第二个内核**。

## 为什么并存而不是替换

实测（V100，BH=72、S=2048、dh=64，前向+反向）：
    ｜ 原版（Q/dO 常驻 68 KB）｜ 0.044 s
    ｜ 窗口版（Q/dO 滑窗 40.5 KB）｜ 0.059 s  ← 慢 34%

规则④要求 V100 三档不得退步，而窗口化要多读 Q/dO，必然变慢。
所以：**按设备动态 smem 上限选内核**
  · 上限 ≥ 68 KB（V100/sm_70）→ 原版 `attn_bwd_flash`（快）
  · 上限 < 68 KB（P100/sm_60）→ 窗口版 `attn_bwd_flash_w`（40.5 KB，能编）

## 做法

从实验场 `lab/attn32/kernel.cu`（已验证逐位/1e-7 一致）里：
1. 取 `load_win` 与 `flash_qk_dp_win` 两个新 helper
2. 用大括号配对抽出被改过的 `attn_bwd_flash` 函数体，改名 `attn_bwd_flash_w`
3. 插进生产 `.cu`（原 `attn_bwd_flash` 一行不动）

用法：python tools/port_attn_window_kernel.py
"""
import io
import os

PROD = "python/tgai_go/train/kernels/tgai_attn_nop.cu"
LAB = "lab/attn32/kernel.cu"


def extract_function(src, name):
    """按大括号配对抽出 `name(...)` 这个函数的完整定义（含签名行与结尾 }）。"""
    key = "attn_bwd_flash(" if name == "attn_bwd_flash" else "attn_bwd_flash_w("
    # 找到函数名出现处，往前回退到行首（可能带 __global__ / __launch_bounds__）
    i = src.index("\n" + key) if ("\n" + key) in src else src.index(key)
    start = src.rfind("\n", 0, i) + 1
    # 若前一行还是属性行，继续回退
    while True:
        prev = src.rfind("\n", 0, start - 1) + 1
        line = src[prev:start]
        if "__launch_bounds__" in line or line.strip() == "":
            start = prev
            if "__launch_bounds__" in line:
                continue
        break
    open_brace = src.index("{", i)
    depth = 0
    j = open_brace
    while j < len(src):
        c = src[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return start, j + 1, src[start:j + 1]
        j += 1
    raise SystemExit("大括号没配平：%s" % name)


def main():
    prod = io.open(PROD, encoding="utf-8").read()
    lab = io.open(LAB, encoding="utf-8").read()

    if "attn_bwd_flash_w" in prod:
        print("生产里已有 attn_bwd_flash_w，跳过")
        return 0

    # ---- 1) 抽 helper（load_win 与 flash_qk_dp_win）----
    a = lab.index("// ── 窗口装载")
    b = lab.index("__device__ __forceinline__ void load_T72(")
    helpers = lab[a:b]
    assert "load_win" in helpers and "flash_qk_dp_win" in helpers

    # ---- 2) 抽窗口版反向内核 ----
    s0, s1, body = extract_function(lab, "attn_bwd_flash")
    assert "atomicAdd(dkrow" in body, "抽出来的函数体不对"
    assert "load_win(" in body, "抽出来的函数体不是窗口版"
    body_w = body.replace("attn_bwd_flash(", "attn_bwd_flash_w(", 1)
    assert "attn_bwd_flash_w(" in body_w

    # ---- 3) 插进生产 ----
    # 3a) DCHUNK 定义
    assert "#define DCHUNK" not in prod
    prod = prod.replace("#define BT 64",
                        "#define BT 64\n"
                        "#define DCHUNK 16   // head_dim 窗口宽度（P100 变体用；CUTLASS: ThreadK*2）",
                        1)

    # 3b) helper 插在 load_T72 之前
    prod = prod.replace("__device__ __forceinline__ void load_T72(",
                        helpers + "__device__ __forceinline__ void load_T72(", 1)

    # 3c) 窗口版内核插在原内核之后（原内核结束的下一个 "// One query tile" 注释前）
    marker = "// One query tile against the K/V already in smem."
    assert marker in prod
    assert body_w not in prod
    banner = ("\n// ── P100（sm_60，动态 smem 上限 48 KB）专用变体：Q/dO 沿 head_dim 滑窗，\n"
              "//    动态 smem 41472 B（40.5 KB）。数值与上游 attn_bwd_flash 相差 < 1.1e-7（相对），\n"
              "//    y 逐位相同。V100 仍走上游版本（快 34%）。见 lab/attn32/LOG.md。\n")
    prod = prod.replace(marker, banner + body_w + "\n\n" + marker, 1)

    io.open(PROD, "w", encoding="utf-8", newline="\n").write(prod)
    print("已把窗口版内核写入生产：attn_bwd_flash_w（原内核未动）")
    print("helper: load_win / flash_qk_dp_win；DCHUNK=16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
