# -*- coding: utf-8 -*-
"""Would cuBLAS TF32 (tensor cores, **zero casts**) beat --amp off? (ASCII only)

`--amp gemm` pays ~985 cast launches (~11.6 ms/step) to reach tensor cores.
TF32 reaches the same tensor cores by changing only the cuBLAS math mode, so it
costs 0 launches and 0 bytes of conversion traffic.  This probe wraps the whole
training in `ns_math_mode_begin/end` (the existing Muon helper) with amp off.
Numerics: TF32 rounds inputs to a 10-bit mantissa -> NOT bit-identical to fp32.
"""
import atexit
import json
import os
import sys

OUT = os.environ.get("PROBE_OUT", "/root/tf32.json")

import tgai_go.train as T          # noqa: E402
from tgai_go.train import v3_pretrain as VP   # noqa: E402

rt = T._cuda()
st = {"begun": False}

_begin = rt.ns_math_mode_begin
_end = rt.ns_math_mode_end
_orig_main = VP.main


def main(argv=None):
    ok = _begin()
    st["begun"] = bool(ok)
    sys.stderr.write("[tf32] math mode begin -> %r\n" % (ok,))
    try:
        return _orig_main(argv)
    finally:
        try:
            _end()
        except Exception:      # noqa: BLE001
            pass


VP.main = main


def dump():
    try:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(st, f)
    except Exception:          # noqa: BLE001
        pass


atexit.register(dump)
raise SystemExit(VP.main(sys.argv[1:]))
