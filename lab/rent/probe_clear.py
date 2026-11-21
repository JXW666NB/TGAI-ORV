# -*- coding: utf-8 -*-
"""Who calls f16_cache_clear(), and when? (ASCII only)"""
import atexit
import collections
import json
import os
import sys

OUT = os.environ.get("PROBE_OUT", "/root/probe_clear.json")

import tgai_go.train as T          # noqa: E402
from tgai_go.train import v3_pretrain as VP   # noqa: E402

rt = T._cuda()
C = collections.Counter()
sites = collections.Counter()
seq = []          # ordered list of (mark_index, site, entries)
mark_i = [0]
orig_clip = T.clip_grad_norm_


def clip(*a, **kw):
    mark_i[0] += 1
    return orig_clip(*a, **kw)


T.clip_grad_norm_ = clip


def site():
    f = sys._getframe(2)
    out = []
    while f is not None and len(out) < 3:
        fn = f.f_code.co_filename
        b = os.path.basename(fn)
        if "tgai_go" in fn and (b.startswith("v3_") or b in
                                ("amp_api.py", "int8_adamw_gpu.py", "opt_fp16mv.py",
                                 "opt_extra.py", "__init__.py")):
            out.append("%s:%d:%s" % (b, f.f_lineno, f.f_code.co_name))
        f = f.f_back
    return " <- ".join(out) if out else "?"


_clear = rt.f16_cache_clear


def clear():
    n = len(getattr(rt, "_f16_cache", None) or {})
    s = site()
    sites[(s, n)] += 1
    seq.append([mark_i[0], s, n])
    C["calls"] += 1
    C["entries"] += n
    return _clear()


rt.f16_cache_clear = clear


def dump():
    try:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({"counts": dict(C), "sites": [[list(k), v] for k, v in
                                                   sorted(sites.items(), key=lambda kv: -kv[1])],
                       "seq": seq[:80]}, f, ensure_ascii=False, indent=1)
        sys.stderr.write("[probe] wrote %s\n" % OUT)
    except Exception as e:      # noqa: BLE001
        sys.stderr.write("[probe] dump failed %r\n" % (e,))


atexit.register(dump)
raise SystemExit(VP.main(sys.argv[1:]))
