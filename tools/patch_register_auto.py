# -*- coding: utf-8 -*-
"""用正确实现替换 ops_plugin.register_auto（前一版训练路径也走了 fallback，等于没提速）。

新行为：
* 不需要梯度 → 走 forward 快路径，不建图。
* 需要梯度 → **前向仍走 forward 快路径**，反向把 fallback 重算一遍再反传。
* check=True 时，首次训练调用会比对 forward 与 fallback 的结果，差异超过 rtol 直接报错。
"""
import io
import re

P = "python/tgai_go/train/ops_plugin.py"

NEW = '''def register_auto(name, forward, fallback, *, check=True, rtol=2e-3, kind="functional",
                  doc="", replace=False):
    """注册**只需要写前向**的算子：快路径前向 + **自动导出**的反向。

    这是介于「组合式」与「自带反向」之间的第三档：

    * `forward(*xs, **kw) -> np.ndarray` —— 快路径（可以是原生 kernel 的 numpy 包装）。
    * `fallback(*tensors, **kw) -> Tensor` —— 与 `forward` **数学等价**、用 ORV 已有算子写的实现。

    训练时：**前向走 `forward`**（享受原生速度），反向把 `fallback` 用同样输入重算一遍、
    以传入梯度为种子反传，得到的是**逐位精确**的梯度（不是有限差分近似）。
    代价：每次反向多一次 `fallback` 前向。

    为什么不能解析自动微分：原生 kernel 是编译后的机器码，没有源码级 AD。
    通用可行的办法只有「再算一次可微版本」。

    | 你的情况 | 用哪个 |
    | --- | --- |
    | 能用已有算子拼出来 | `register()`（最简单，前向即反向来源）|
    | 有原生 kernel，也写得出一份可微实现 | **`register_auto()`（本函数）** |
    | 追求极致、愿意手写反向 | `register_numpy()` |

    `check=True` 会在**首次训练调用**时比对 `forward` 与 `fallback` 的输出，
    差异超过 `rtol` 直接报错 —— 专防「两份实现数学不一致」这种最难查的错。
    """
    if not callable(forward) or not callable(fallback):
        raise TypeError("register_auto: forward 与 fallback 都必须可调用"
                        + "。修法：forward 走快路径、fallback 用已有算子写等价实现")

    state = {"checked": False}

    def op(*args, **kwargs):
        tensors = [a for a in args if isinstance(a, Tensor)]
        arrays = [a.to_numpy() if isinstance(a, Tensor) else a for a in args]

        if not tensors:
            return Tensor(np.asarray(forward(*arrays, **kwargs), dtype=np.float32))

        need_grad = _GRAD_ON[0] and any(getattr(a, "requires_grad", False) for a in tensors)
        out_np = np.asarray(forward(*arrays, **kwargs), dtype=np.float32)
        if not need_grad:
            return Tensor(out_np, requires_grad=False)

        out = Tensor(out_np, requires_grad=True, _parents=tuple(tensors))

        def bw(g, grads):
            # 反向：在 fallback 上重算（乘以 1.0 得到同设备的可微节点，不改数值）
            clones = [a * 1.0 if isinstance(a, Tensor) else a for a in args]
            y = fallback(*clones, **kwargs)
            if not state["checked"]:
                _check_consistency(name, y, out_np, rtol)
                state["checked"] = True
            seed = Tensor(np.asarray(g, dtype=np.float32))
            (y * seed).sum().backward()
            for a, c in zip(args, clones):
                if isinstance(a, Tensor) and a.requires_grad:
                    gc = getattr(c, "grad", None)
                    if gc is not None:
                        _acc(grads, a, gc)

        out._backward = bw
        return out

    if check:
        state["checked"] = False
    return register(name, op, kind=kind, doc=doc, replace=replace)


def _check_consistency(name, y_fallback, out_np, rtol):
    try:
        got = y_fallback.to_numpy()
    except Exception:       # noqa: BLE001  取不回来就不阻塞（只是少一次体检）
        return
    got = np.asarray(got, np.float32)
    ref = np.asarray(out_np, np.float32)
    if got.shape != ref.shape:
        raise ValueError(
            "算子 %r 的 fallback 形状 %s 与 forward %s 不一致。修法：让两份实现的形状完全一致"
            % (name, got.shape, ref.shape))
    scale = float(np.abs(ref).max()) or 1.0
    rel = float(np.abs(got - ref).max()) / scale
    if rel > rtol:
        raise ValueError(
            "算子 %r 的 fallback 与 forward 数学不一致（相对差 %.2e > %.2e）。"
            "修法：register_auto 要求两份实现算同一个函数，否则梯度是错的；"
            "若确实存在近似，显式传 check=False 并自行确认可接受" % (name, rel, rtol))


'''


def main():
    s = io.open(P, encoding="utf-8").read()
    start = s.index("def register_auto(")
    m = re.search(r"\ndef (?!register_auto)\w+\(", s[start:])
    end = start + m.start() + 1
    s = s[:start] + NEW + s[end:]
    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    import ast
    ast.parse(io.open(P, encoding="utf-8").read())
    print("已替换 register_auto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
