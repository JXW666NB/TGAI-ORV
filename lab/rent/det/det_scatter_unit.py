# -*- coding: utf-8 -*-
"""确定性散射内核的单元自检（GPU，秒级）。

检查：
  1. 新内核能编过（NVRTC 在第一次 launch 时编译，语法错会在这里炸）
  2. `rt.deterministic=True` 时 `scatter_add_rows` / `scatter_add_flat`
     两次调用**逐位相同**
  3. 新内核的结果与 numpy `np.add.at`（升序串行）**逐位相同** ——
     不只是"接近"，因为顺序本来就是同一个
  4. 关闭时仍走原子版（记录它与 numpy 的差异量级，作为"默认路径没被改动"的旁证）
  5. `TG_DETERMINISTIC` 的非法值在**子进程**里明确报错、合法值正确解析
     （`cuda.py` 被 exec 进私有命名空间，模块级函数外部拿不到 ⇒ 用子进程验）

跑法：D:\\orv_bench\\remote_py.bat D:\\orv_bench\\det_scatter_unit.py D:\\orv_det_work\\new_pkg
"""
import json
import os
import subprocess
import sys

PKG = sys.argv[1] if len(sys.argv) > 1 else ""
if PKG:
    sys.path.insert(0, PKG)

import numpy as np

import tgai_go.train as T

OUT = r"D:\orv_bench\det_scatter_unit.json"


def env_case(val):
    """在子进程里用一个 env 值建 rt，返回 (rc, 是否开了, 输出)。"""
    code = ("import tgai_go.train as T;rt=T._cuda();"
            "print('FLAG=%r' % (rt.deterministic,))")
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if PKG:
        env["PYTHONPATH"] = PKG
    env["TG_DETERMINISTIC"] = val
    p = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    text = (p.stdout or "") + (p.stderr or "")
    return p.returncode, text


def main():
    rt = T._cuda()
    rng = np.random.default_rng(11)
    res = {}

    # ---- 行散射：n_src=1000, n_rows=300, row_size=16，重复下标很多 ----
    n_src, row_size, n_rows = 1000, 16, 300
    idx = rng.integers(0, n_src, n_rows).astype(np.int32)
    g = rng.standard_normal((n_rows, row_size)).astype(np.float32)
    ref = np.zeros((n_src, row_size), np.float32)
    for r in range(n_rows):
        ref[idx[r]] += g[r]
    gbuf = rt.h2d(g)
    ibuf = rt.h2d(idx, dtype=np.int32)
    old = getattr(rt, "deterministic", False)
    try:
        rt.deterministic = True
        a = rt.d2h(rt.scatter_add_rows(gbuf, ibuf, n_src, row_size, n_rows)).copy()
        b = rt.d2h(rt.scatter_add_rows(gbuf, ibuf, n_src, row_size, n_rows)).copy()
        res["rows_det_repeat_equal"] = bool(np.array_equal(a, b))
        res["rows_det_equals_numpy_bitwise"] = bool(np.array_equal(a, ref))
        res["rows_det_max_abs_diff_vs_numpy"] = float(np.abs(a - ref).max())
        rt.deterministic = False
        c = rt.d2h(rt.scatter_add_rows(gbuf, ibuf, n_src, row_size, n_rows)).copy()
        res["rows_atomic_matches_numpy_bitwise"] = bool(np.array_equal(c, ref))
        res["rows_atomic_max_abs_diff_vs_numpy"] = float(np.abs(c - ref).max())
        res["rows_atomic_frac_diff_vs_numpy"] = float(np.mean(c != ref))
    finally:
        rt.deterministic = old

    # ---- 逐元素散射 ----
    n, n_src_f = 300, 1000
    idxf = rng.integers(0, n_src_f, n).astype(np.int32)
    gf = rng.standard_normal(n).astype(np.float32)
    reff = np.zeros(n_src_f, np.float32)
    for i in range(n):
        reff[idxf[i]] += gf[i]
    gfb = rt.h2d(gf)
    ifb = rt.h2d(idxf, dtype=np.int32)
    try:
        rt.deterministic = True
        a = rt.d2h(rt.scatter_add_flat(gfb, ifb, n_src_f, n)).copy()
        b = rt.d2h(rt.scatter_add_flat(gfb, ifb, n_src_f, n)).copy()
        res["flat_det_repeat_equal"] = bool(np.array_equal(a, b))
        res["flat_det_equals_numpy_bitwise"] = bool(np.array_equal(a, reff))
        rt.deterministic = False
        c = rt.d2h(rt.scatter_add_flat(gfb, ifb, n_src_f, n)).copy()
        res["flat_atomic_max_abs_diff_vs_numpy"] = float(np.abs(c - reff).max())
    finally:
        rt.deterministic = old

    # ---- 环境变量解析（子进程）----
    res["env"] = {}
    for val in ("", "0", "1", "true", "off", "YES"):
        rc, txt = env_case(val)
        ok = (rc == 0 and "FLAG=True" in txt) if val in ("1", "true", "YES") else \
             (rc == 0 and "FLAG=False" in txt)
        res["env"][val if val else "<empty>"] = {"rc": rc, "ok": ok}
    rc, txt = env_case("banana")
    res["env"]["banana"] = {"rc": rc, "rejected": rc != 0,
                            "msg": [ln for ln in txt.splitlines()
                                    if "TG_DETERMINISTIC" in ln][:2]}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for k, v in res.items():
        print("  %-38s %s" % (k, v))
    ok = (res["rows_det_repeat_equal"] and res["rows_det_equals_numpy_bitwise"]
          and res["flat_det_repeat_equal"] and res["flat_det_equals_numpy_bitwise"]
          and res["env"]["banana"]["rejected"]
          and all(v.get("ok") for k, v in res["env"].items() if k != "banana"))
    print("[det_scatter_unit] %s -> %s" % ("PASS" if ok else "FAIL", OUT))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
