# -*- coding: utf-8 -*-
"""修引擎 BUG：惰性转置 `_lazy_T` 被绝大多数算子忽略，静默给出未转置的数据。

## 症状（实测，V100）

    x = from_numpy(arange(6).reshape(3,2)); t = x.T
    t.to_numpy()      -> 正确（to_numpy 认 _lazy_T）
    t.reshape(6)      -> [0,1,2,3,4,5]   应为 [0,2,4,1,3,5]
    t + 1 / t * 2 / -t / t.exp()  -> 全部按未转置缓冲算
    t.sum(axis=1)     -> 连形状都不对
    t.T               -> 不等于 x

## 根因

`_transpose_cuda`（`__init__.py:1128`）不物化，只把 `_lazy_T` 标记打在结果上；
只有 `matmul`（走 `gemm_nt`）与 `to_numpy` 认这个标记，其余算子直接读 `self._gpu`。

## 修法（单一入口）

把 `_gpu` 变成**属性**：getter 发现 `_lazy_T` 就先就地物化，再返回缓冲。
`matmul` 的 `gemm_nt` 快路径读的是**源张量** `src._gpu`（源永远不是惰性的），
所以省拷贝的优化不受影响。

物化是**就地**的：把 `_gpu_raw` 换成 `rt.transpose2d(base)` 的结果，清掉
`_lazy_T`/`_lazy_shape`；`_backward` 本来就写的是 `transpose2d(g)`，图不用重建。

用法：python tools/fix_lazy_transpose.py
"""
import io

P = "python/tgai_go/train/__init__.py"

SLOTS_OLD = '''    __slots__ = ("data", "grad", "requires_grad", "_parents", "_bwd", "device", "_gpu",
                 "_lazy_T", "_lazy_shape")'''
SLOTS_NEW = '''    __slots__ = ("data", "grad", "requires_grad", "_parents", "_bwd", "device", "_gpu_raw",
                 "_lazy_T", "_lazy_shape")'''

PROP = '''    # ---- 惰性转置的统一出口 -------------------------------------------
    # `_transpose_cuda` 不物化，只打 `_lazy_T` 标记（为 2D matmul 的 gemm_nt 省一次拷贝）。
    # 但除 matmul / to_numpy 之外的算子都直接读缓冲，会静默拿到**未转置**的数据。
    # 所以把 `_gpu` 做成属性：谁读缓冲，谁就先触发物化。
    @property
    def _gpu(self):
        if self._lazy_T is not None:
            self._materialize_inplace()
        return self._gpu_raw

    @_gpu.setter
    def _gpu(self, value):
        self._gpu_raw = value

    def _materialize_inplace(self):
        """把本张量从"惰性转置视图"变成真正的转置数据（就地，图不变）。"""
        rt = _cuda()
        base = self._lazy_T
        if base is None:
            return self
        self._gpu_raw = rt.transpose2d(base._gpu)
        self._lazy_T = None
        self._lazy_shape = None
        return self

'''


def main():
    s = io.open(P, encoding="utf-8").read()

    if "def _materialize_inplace" in s:
        print("已经打过补丁")
        return 0

    assert SLOTS_OLD in s, "找不到 __slots__ 锚点"
    s = s.replace(SLOTS_OLD, SLOTS_NEW, 1)

    anchor = "    def _matmul_cuda_nd(self, other):"
    assert anchor in s, "找不到 _matmul_cuda_nd 锚点"
    s = s.replace(anchor, PROP + anchor, 1)

    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    print("已改：__slots__ 的 _gpu -> _gpu_raw；新增 _gpu 属性 + _materialize_inplace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
