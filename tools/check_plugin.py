# -*- coding: utf-8 -*-
"""插件自检：契约是否合规 + **数值梯度是否真的对**。

为什么需要它：插件是用户代码，引擎不审计它的数学。而"反向写错"是最常见的错误，
且**不会报错** —— 训练照样跑、loss 照样降，只是学得不对。所以必须能一条命令验：

    python tools/check_plugin.py examples/plugins/squared_relu.py

检查项：
  1. 能否加载（语法/导入）；
  2. `PLUGIN` 契约是否合规（name/version/api_version/ops）；
  3. 每个声明了 `test_inputs` 的算子：
     * 前向能跑、输出有限；
     * 若可求导 → 用**中心差分**核对 autograd 的梯度（相对误差 <= tol）。

判定阈值说明：引擎是 float32，中心差分本身有 ~1e-3 的截断/舍入误差，
所以默认 tol=2e-2。结构性错误（少个转置、符号反了）是 O(1) 量级，这个阈值足够抓住。
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

import tgai_go.train as T                     # noqa: E402
from tgai_go.train import ops_plugin as P      # noqa: E402


def _make_inputs(specs):
    """按 test_inputs 规格造 (tensor, requires_grad) 列表。"""
    out = []
    for s in specs:
        s = dict(s or {})
        if "value" in s:
            arr = np.asarray(s["value"], dtype=np.float32)
        else:
            shape = tuple(int(x) for x in s.get("shape") or (2, 4))
            lo = float(s.get("low", -2.0))
            hi = float(s.get("high", 2.0))
            rng = np.random.default_rng(int(s.get("seed", 0)))
            arr = rng.uniform(lo, hi, size=shape).astype(np.float32)
        out.append((arr, bool(s.get("requires_grad", True))))
    return out


def _scalar_loss(tensors, fn):
    ts = [T.tensor(a.copy(), requires_grad=rg) for a, rg in tensors]
    out = fn(*ts)
    loss = (out * out).sum()
    return out, loss, ts


def check_op(name, fn, specs, eps=1e-2, tol=2e-2):
    """返回 (ok, 说明)。用 sum(out^2) 当标量损失，因此与输出形状无关。"""
    tensors = _make_inputs(specs)
    out, loss, ts = _scalar_loss(tensors, fn)
    arr = np.asarray(out.to_numpy() if hasattr(out, "to_numpy") else out, np.float32)
    if not np.isfinite(arr).all():
        return False, "前向输出含 NaN/Inf"

    grad_ts = [i for i, (_, rg) in enumerate(tensors) if rg]
    if not grad_ts:
        return True, "前向 OK（无输入需要梯度）"
    try:
        loss.backward()
    except Exception as e:                       # noqa: BLE001
        return False, "反向抛异常：%s: %s" % (type(e).__name__, str(e)[:120])

    worst = None
    for idx in grad_ts:
        g = ts[idx].grad
        if g is None:
            return False, "输入 %d 需要梯度但 grad 是 None" % idx
        ga = np.asarray(g.to_numpy() if hasattr(g, "to_numpy") else g, np.float64)
        # 中心差分：同一份输入，扰动第 idx 个
        base = [a.astype(np.float64) for a, _ in tensors]
        fd = np.zeros_like(base[idx])
        it = np.ndindex(*base[idx].shape)
        for pos in it:
            for sign in (+1, -1):
                probe = [a.copy() for a in base]
                probe[idx][pos] += sign * eps
                ts2 = [T.tensor(a.astype(np.float32),
                                requires_grad=False) for a in probe]
                o2 = fn(*ts2)
                v2 = float((np.asarray(o2.to_numpy() if hasattr(o2, "to_numpy") else o2,
                                       np.float64) ** 2).sum())
                fd[pos] += sign * v2 / (2.0 * eps)
        scale = max(float(np.abs(fd).max()), float(np.abs(ga).max()), 1e-9)
        rel = float(np.abs(ga - fd).max()) / scale
        if worst is None or rel > worst[1]:
            worst = (idx, rel)

    if worst and worst[1] > tol:
        return False, ("输入 %d 的梯度和中心差分差 %.2e（阈值 %.0e）。"
                       "常见原因：反向的转置方向、sum 的维度、少了/多了系数"
                       % (worst[0], worst[1], tol))
    return True, "前向+梯度 OK（最大相对误差 %.2e）" % (worst[1] if worst else 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("plugin", help="插件 .py 文件")
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    T.set_default_device("cpu")
    path = os.path.abspath(a.plugin)
    report = {"plugin": path, "load": None, "contract": [], "ops": [], "ok": True}

    try:
        mod = P.load_plugin_file(path)
        report["load"] = "OK"
    except Exception as e:                        # noqa: BLE001
        report["load"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        report["ok"] = False
        print(json.dumps(report, ensure_ascii=False, indent=2) if a.json
              else "加载失败：" + report["load"])
        return 2

    info = getattr(mod, "PLUGIN", None) or {}
    rec = P.manifest()[-1]
    report["contract"] = rec.get("contract_problems") or []
    if report["contract"]:
        report["ok"] = False

    ops = info.get("ops") or {}
    for name, spec in ops.items():
        fn = getattr(T.functional, name, None)
        if fn is None:
            report["ops"].append({"op": name, "ok": False, "detail": "functional 里找不到"})
            report["ok"] = False
            continue
        specs = (spec or {}).get("test_inputs")
        if not specs:
            report["ops"].append({"op": name, "ok": True,
                                  "detail": "没有 test_inputs，只确认已注册"})
            continue
        try:
            ok, detail = check_op(name, fn, specs, tol=a.tol)
        except Exception as e:                    # noqa: BLE001
            ok, detail = False, "%s: %s" % (type(e).__name__, str(e)[:160])
        report["ops"].append({"op": name, "ok": ok, "detail": detail})
        report["ok"] = report["ok"] and ok

    if a.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("插件：%s" % os.path.basename(path))
        print("  加载：%s" % report["load"])
        print("  API：%s（引擎 %s）  版本：%s  指纹：%s"
              % (info.get("api_version"), P.API_VERSION, info.get("version"),
                 rec.get("sha256", "")[:16] + "…"))
        if report["contract"]:
            print("  契约问题：")
            for pb in report["contract"]:
                print("    - %s" % pb)
        for r in report["ops"]:
            print("  [%s] %-18s %s" % ("OK " if r["ok"] else "FAIL", r["op"], r["detail"]))
        print("结论：%s" % ("PASS" if report["ok"] else "FAIL"))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
