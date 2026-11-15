# -*- coding: utf-8 -*-
"""让融合反向在设置不了动态 smem 的卡上**事前就被识别**，直接走物化路径。

背景（2026-11-15 真机 P100）：`attn_bwd_flash_w` 的动态 smem 属性设不上（rc=1），
而失败发生在**反向**，前向捕获不到 ⇒ 训练步在 backward 里硬崩。
修法：在选择 lowmem 之前做一次**探针**（试着设一次属性并缓存结果），设不上就不选 lowmem。
"""
import io

P_CUDA = "python/tgai_go/train/cuda.py"
P_INIT = "python/tgai_go/train/__init__.py"

PROBE = '''
    def attn_flash_bwd_supported(self):
        """本卡能否为**融合反向**设上所需的动态 smem（结果缓存，只探一次）。

        为什么要事前探：属性设置失败发生在 backward 的内核启动时，前向看不出问题，
        于是整步在 backward 里硬崩（真机 P100 就是这样）。先试一次，就能在**选择路径**
        的时候决定要不要走融合，避免半路炸。
        """
        cached = getattr(self, "_attn_flash_bwd_ok", None)
        if cached is not None:
            return bool(cached)
        ok = True
        try:
            self.ensure_attn_nop()
            lim = int(self._device_smem_limit() or 0)
            if lim and _ATTN_FLASH_WIN_SMEM <= lim < _ATTN_FLASH_SMEM:
                self._attn_set_dyn_smem("attn_bwd_flash_w", _ATTN_FLASH_WIN_SMEM)
            else:
                self._attn_set_dyn_smem("attn_bwd_flash", _ATTN_FLASH_SMEM)
        except Exception:            # noqa: BLE001  探测失败就当不支持，走物化（正确性优先）
            ok = False
        self._attn_flash_bwd_ok = ok
        return ok

'''

OLD_OK = '''    bh, seq, _dh = q.shape
    return int(bh) > 0 and int(seq) > 0'''

NEW_OK = '''    bh, seq, _dh = q.shape
    if int(bh) <= 0 or int(seq) <= 0:
        return False
    # 本卡若设不上融合反向需要的动态 smem（如 P100 / sm_60），现在就改走物化，
    # 而不是等到 backward 里 rc=1 把整个训练步炸掉。
    try:
        if not _cuda().attn_flash_bwd_supported():
            global _ATTN_FLASH_WARNED
            if not _ATTN_FLASH_WARNED:
                _ATTN_FLASH_WARNED = True
                import warnings as _w
                _w.warn("本卡的融合注意力反向用不了（动态共享内存设不上），已自动改用"
                        "物化注意力：结果正确，但显存与速度都会差一些。"
                        "想要融合路径请换 V100（sm_70）及以上。", RuntimeWarning)
            return False
    except Exception:            # noqa: BLE001  探测本身出错也不能影响正确性
        return False
    return True'''

WARN_DECL = "_ATTN_FLASH_WARNED = False\n\n\n"


def main():
    s = io.open(P_CUDA, encoding="utf-8").read()
    if "def attn_flash_bwd_supported" in s:
        print("cuda.py 已有探针")
    else:
        anchor = "    def _attn_set_dyn_smem(self, name, smem):"
        assert anchor in s
        s = s.replace(anchor, PROBE.lstrip("\n") + anchor, 1)
        io.open(P_CUDA, "w", encoding="utf-8", newline="\n").write(s)
        print("cuda.py 已加 attn_flash_bwd_supported")

    t = io.open(P_INIT, encoding="utf-8").read()
    if "_ATTN_FLASH_WARNED" in t:
        print("__init__.py 已改")
        return 0
    assert OLD_OK in t, "找不到 _sdpa_lowmem_ok 尾部锚点"
    t = t.replace(OLD_OK, NEW_OK, 1)
    anchor2 = "def _sdpa_lowmem_ok(q, k, v, causal, dh):"
    t = t.replace(anchor2, WARN_DECL + anchor2, 1)
    io.open(P_INIT, "w", encoding="utf-8", newline="\n").write(t)
    import ast
    ast.parse(io.open(P_INIT, encoding="utf-8").read())
    print("__init__.py 已加事前探测 + 一次性警告")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
