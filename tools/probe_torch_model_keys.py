# -*- coding: utf-8 -*-
"""确认 `model.py` 的 `TGAILanguageModel` 与 `.pt` 的 `model_state_dict` **键名一致**。

用 `torch.device("meta")` 建模型：只跑形状推导，**不分配内存**，所以 2.999B 也能秒过。
这一步的目的是在"停产线开窗"之前就把 torch 侧最大的不确定性消掉。
"""
import importlib.util
import json
import os
import sys

import torch


def load_model_module(path):
    spec = importlib.util.spec_from_file_location("tgai_ref_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    # 路径一律从配置文件读：远端用 ssh 传带空格/引号的路径会被 shell 吃掉
    # （实测 `"D:/.../TGAI NB/..."` 会变成 `'C:\Users\Administrator\ D:/...'`）。
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "bench_cfg.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg_json = json.load(f)
    pt = cfg_json["model_pt"]
    model_py = cfg_json["model_py"]

    ck = torch.load(pt, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["model_state_dict"]
    mod = load_model_module(model_py)

    cfg = mod.TGAIConfig(**cfg_json.get("config", {}))
    # 用 .pt 里的实际形状推关键字段，避免依赖 config 默认值
    out = {"n_state_dict_keys": len(sd)}

    # TGAIConfig 的字段清单（dataclass 的 __annotations__）
    out["config_fields"] = sorted(getattr(mod.TGAIConfig, "__annotations__", {}).keys())

    with torch.device("meta"):
        try:
            model = mod.TGAILanguageModel(cfg)
        except Exception as exc:                              # noqa: BLE001
            out["build_error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
            print(json.dumps(out, ensure_ascii=False, indent=1))
            return 1

    mk = set(model.state_dict().keys())
    pk = set(sd.keys())
    out["n_model_keys"] = len(mk)
    out["missing_in_model"] = sorted(pk - mk)[:10]
    out["extra_in_model"] = sorted(mk - pk)[:10]
    out["n_missing"] = len(pk - mk)
    out["n_extra"] = len(mk - pk)
    out["exact_match"] = (mk == pk)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if mk == pk else 2


if __name__ == "__main__":
    sys.exit(main())
