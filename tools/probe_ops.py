# -*- coding: utf-8 -*-
"""极短探针：装包后打印命名空间里有哪些算子。"""
import sys

print("[a] start", flush=True)
sys.path.insert(0, "D:/orv_pip_0962")
sys.path.insert(0, "D:/orv_pip_0962/Lib/site-packages")

import tgai_go.train as T  # noqa: E402

T.plugin.install("examples/bundles/svs-ops", replace=True)
print("[b] ops =", sorted(n for n in dir(T.svs) if not n.startswith("_")), flush=True)
print("[c] version =", T.plugin.info("svs-ops")["version"], flush=True)
