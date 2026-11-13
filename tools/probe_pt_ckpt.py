# -*- coding: utf-8 -*-
"""看一眼 `.pt` 检查点的结构：键名、是否包着 `state_dict`、参数量。

只读，不建模型、不进 GPU（`map_location="cpu"` + `mmap=True`，不把 6 GB 读进内存）。
"""
import json
import sys

import torch


def main():
    path = sys.argv[1]
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    out = {"file": path, "top_type": type(sd).__name__}
    if isinstance(sd, dict):
        keys = list(sd.keys())
        out["n_top"] = len(keys)
        out["top_sample"] = keys[:8]
        # 常见包装层
        for wrap in ("model", "state_dict", "module", "model_state_dict"):
            if wrap in sd and isinstance(sd[wrap], dict):
                out["wrapper"] = wrap
                inner = sd[wrap]
                ik = list(inner.keys())
                out["n_inner"] = len(ik)
                out["inner_sample"] = ik[:10]
                # 参数量
                tot = 0
                for v in inner.values():
                    if hasattr(v, "numel"):
                        tot += int(v.numel())
                out["params_b"] = round(tot / 1e9, 3)
                dtypes = {}
                for v in inner.values():
                    if hasattr(v, "dtype"):
                        dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + 1
                out["dtypes"] = dtypes
                break
        else:
            tot = 0
            for v in sd.values():
                if hasattr(v, "numel"):
                    tot += int(v.numel())
            out["params_b"] = round(tot / 1e9, 3)
            out["has_config"] = [k for k in keys
                                 if not hasattr(sd[k], "numel")][:12]
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
