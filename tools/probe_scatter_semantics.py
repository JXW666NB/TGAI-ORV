# -*- coding: utf-8 -*-
"""在真机上钉住 `gather_rows` / `scatter_add_rows` 的参数语义。

为什么必须先测：这两个运行时的签名是
    gather_rows(src, idx, n_src_rows, row_size, n_rows)
    scatter_add_rows(g, idx, n_src_rows, row_size, n_rows)
`n_src_rows` / `n_rows` 哪个是"源行数"、哪个是"目标行数"无法从名字确定，
猜错会得到一个形状正确但语义相反的算子 —— 那类错误在训练里表现为"loss 还是降的"。

这个脚本用**已知数值**跑一遍，打印结果，让人一眼判定语义。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import tgai_go.train as T  # noqa: E402


def show(name, buf, rows, d):
    arr = buf.to_numpy() if hasattr(buf, "to_numpy") else buf
    a = np.asarray(arr, np.float32).reshape(rows, d)
    print("%-18s shape=(%d,%d)\n%s" % (name, rows, d, a))


def main():
    rt = T._cuda()
    T.set_default_device("cuda")
    d = 3
    table = np.array([[10., 11., 12.],
                      [20., 21., 22.],
                      [30., 31., 32.],
                      [40., 41., 42.]], np.float32)         # 4 行"词表"
    ids = np.array([0, 2, 2], np.int32)                       # 取 3 行，其中 2 重复

    src = rt.h2d(np.ascontiguousarray(table))
    ib = rt.h2d(np.ascontiguousarray(ids), dtype=np.int32)

    print("=== gather_rows：期望 [row0, row2, row2] ===")
    try:
        g = rt.gather_rows(src, ib, 4, d, 3)
        show("gather", g, 3, d)
    except Exception as e:
        print("gather ERR", type(e).__name__, str(e)[:160])

    print("\n=== scatter_add_rows：期望 out[0]+=v0, out[2]+=v1+v2（4 行目标）===")
    vals = np.array([[1., 1., 1.],
                     [2., 2., 2.],
                     [4., 4., 4.]], np.float32)              # 3 个源行
    vb = rt.h2d(np.ascontiguousarray(vals))
    for args in ((3, d, 4), (4, d, 3)):
        try:
            out = rt.scatter_add_rows(vb, ib, args[0], args[1], args[2])
            show("scatter%s" % (args,), out, args[2], d)
        except Exception as e:
            print("scatter%s ERR %s %s" % (args, type(e).__name__, str(e)[:120]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
