# -*- coding: utf-8 -*-
"""报告关键第三方库是否可用（用于判断某项目能不能在本机跑）。"""
import importlib
import sys

MODS = ("torch", "torchaudio", "numpy", "scipy", "yaml", "librosa",
        "soundfile", "pyworld", "torchcrepe", "pretty_midi", "onnxruntime")


def main():
    print("python:", sys.version.split()[0], sys.executable)
    for name in MODS:
        try:
            m = importlib.import_module(name)
            print("  %-12s %s" % (name, getattr(m, "__version__", "已安装")))
        except Exception as exc:
            print("  %-12s -- %s" % (name, type(exc).__name__))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
