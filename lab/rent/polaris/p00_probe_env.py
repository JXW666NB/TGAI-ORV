# -*- coding: utf-8 -*-
"""环境探针：ORV 版本 / 设备 API / 优化器清单 / Polaris 可用性。"""
import json
import os
import sys

import numpy as np

import tgai_go
import tgai_go.train as T

out = {}
out["python"] = sys.version
out["exe"] = sys.executable
out["orv_file"] = tgai_go.__file__
out["orv_version"] = getattr(tgai_go, "__version__", None)
try:
    from importlib.metadata import version as _v
    out["orv_dist"] = _v("tgai-orv")
except Exception as e:
    out["orv_dist"] = "ERR:%s" % e

# 设备 API 名称（0.9.x 与 1.x 可能不同）
out["device_api"] = {
    "set_device": hasattr(T, "set_device"),
    "set_default_device": hasattr(T, "set_default_device"),
    "get_device": hasattr(T, "get_device"),
    "device_caps": hasattr(T, "device_caps"),
}
try:
    out["get_device"] = str(T.get_device())
except Exception as e:
    out["get_device"] = "ERR:%s" % e

# 优化器（T.optim 命名空间）
base = [n for n in dir(T.optim) if not n.startswith("_")]
out["T_optim_names"] = base

# 插件
out["plugin_autoload"] = hasattr(T.plugin, "autoload")
out["plugin_registered_before"] = None
try:
    out["plugin_registered_before"] = sorted(T.plugin.registered().get("optimizer", {}) or {})
except Exception as e:
    out["plugin_registered_before"] = "ERR:%s" % e

# Polaris
res = {}
try:
    import polaris
    res["polaris_file"] = polaris.__file__
    res["polaris_version"] = getattr(polaris, "__version__", None)
    res["PLUGIN"] = getattr(polaris, "PLUGIN", None)
    res["resolve"] = str(T.plugin.resolve_optimizer("Polaris"))
except Exception as e:
    res["import_err"] = "%s: %s" % (type(e).__name__, e)
out["polaris"] = res

# 能跑通一个最小 step 吗（CPU）
smoke = {}
try:
    w = T.tensor(np.ones((4, 3), np.float32), requires_grad=True)
    x = T.tensor(np.ones((3, 2), np.float32))
    loss = (T.matmul(w, x)).sum()
    loss.backward()
    smoke["grad_type"] = type(w.grad).__name__
    smoke["grad_shape"] = str(getattr(w.grad, "shape", None))
    smoke["data_type"] = type(w.data).__name__
    opt = T.optim.AdamW([w], lr=1e-2)
    opt.step()
    smoke["adamw_ok"] = True
except Exception as e:
    smoke["adamw_err"] = "%s: %s" % (type(e).__name__, e)
out["smoke"] = smoke

# T.help 是否存在
out["has_help"] = hasattr(T, "help")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT_DIR, exist_ok=True)
_p = os.path.join(OUT_DIR, "p00_probe_env.json")
with open(_p, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=str)
print("WROTE " + _p)
print(json.dumps(out, ensure_ascii=True, indent=1, default=str))
