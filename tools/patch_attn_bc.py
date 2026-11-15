# -*- coding: utf-8 -*-
"""让 attention 反向内核的 key 分块宽 BC 可配置（TG_ATTN_BC），以便 48 KB 卡用常驻版。

背景（实测，见 docs/P100_ATTN_BWD_BOTTLENECK.md）：
  P100 上 attention 反向 82.2 ms vs torch 42.1 ms，×21 层 = 0.84 s，
  占整步差距 0.93 s 的 90%。原因是 68 KB 的常驻内核装不进 48 KB，
  只能退到沿 head_dim 滑窗的版本。

常驻版 smem 公式（已用 45,056 / 40,448 两个实测点反推验证）：
    smem = 4 * 64 * BC * 4   (smK/smV/smP/smDS)
         + 2 * 64 * QSTR * 4 (smQ/smDO, QSTR = DH + 8 = 72)
  BC=32 -> 69,632 B（装不下）
  BC=11 -> 48,128 B（装得下，且不需要滑窗）

改法：把内核里的 `#define BC 32` 换成可配置值，并让两个 smem 常量按 BC 计算。
默认 BC=32，行为与旧版完全一致；设 TG_ATTN_BC=11 才启用窄常驻版。
"""
import io
import re

P = "python/tgai_go/train/cuda.py"

OLD_CONST = """_ATTN_FLASH_SMEM = 69632"""
NEW_CONST = '''# ★ 2026-11-15：key 分块宽 BC 改为可配置（TG_ATTN_BC），以便装不进 96 KB 的卡
#   （如 P100，48 KB）仍能用**常驻**内核而不是滑窗版。公式已用 45,056 / 40,448
#   两个实测点反推验证：
#       smem = 4*64*BC*4 + 2*64*(DH+8)*4
#   BC=32 -> 69,632（默认，与旧版一致）；BC=11 -> 48,128（P100 可用）。
_ATTN_BC = int(os.environ.get("TG_ATTN_BC", "32") or "32")


def _attn_smem_for_bc(bc):
    """常驻反向内核的 smem（字节）。"""
    return 4 * 64 * int(bc) * 4 + 2 * 64 * (64 + 8) * 4


_ATTN_FLASH_SMEM = _attn_smem_for_bc(_ATTN_BC)'''

# 滑窗版 smem 也要跟着 BC 变：2*DCHUNK*QSTR + 4*64*BC
OLD_WIN = "_ATTN_FLASH_WIN_SMEM = 40448"
NEW_WIN = "_ATTN_FLASH_WIN_SMEM = 2 * 8 * 72 * 4 + 4 * 64 * _ATTN_BC * 4"

# 编译时替换宏
OLD_SRC = '''        path = attn_nop_source_path()
        with open(path, encoding="utf-8") as fh:
            src = fh.read()'''
NEW_SRC = '''        path = attn_nop_source_path()
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        if _ATTN_BC != 32:
            # 只在非默认时替换，避免影响既有编译结果
            src = re.sub(r"#define\\s+BC\\s+\\d+", "#define BC %d" % _ATTN_BC, src, count=1)'''


def main():
    s = io.open(P, encoding="utf-8").read()
    if "_attn_smem_for_bc" in s:
        print("已经改过")
        return 0
    assert OLD_CONST in s, "smem 常量锚点失败"
    s = s.replace(OLD_CONST, NEW_CONST, 1)
    assert OLD_WIN in s, "滑窗 smem 锚点失败"
    s = s.replace(OLD_WIN, NEW_WIN, 1)
    assert OLD_SRC in s, "编译锚点失败"
    s = s.replace(OLD_SRC, NEW_SRC, 1)
    if "\nimport re\n" not in s:
        s = s.replace("\nimport os\n", "\nimport os\nimport re\n", 1)
    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    import ast
    ast.parse(io.open(P, encoding="utf-8").read())
    print("已改：TG_ATTN_BC 可配置，smem 按 BC 计算")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
