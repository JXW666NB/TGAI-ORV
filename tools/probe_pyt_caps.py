# -*- coding: utf-8 -*-
"""报告 V100 上 PyTorch 的能力：版本、可选加速库、能否 compile。"""
import sys


def main():
    import torch
    print("torch        :", torch.__version__)
    print("cuda runtime :", torch.version.cuda)
    print("cudnn        :", torch.backends.cudnn.version())
    print("device       :", torch.cuda.get_device_name(0),
          torch.cuda.get_device_capability(0))
    print("cudnn.allow_tf32:", torch.backends.cudnn.allow_tf32,
          " matmul.allow_tf32:", torch.backends.cuda.matmul.allow_tf32)
    for name in ("xformers", "flash_attn", "triton", "liger_kernel", "apex"):
        try:
            mod = __import__(name)
            print("%-12s: %s" % (name, getattr(mod, "__version__", "已安装")))
        except Exception as exc:
            print("%-12s: 未安装 (%s)" % (name, type(exc).__name__))
    # inductor 在 sm70 上能不能跑
    try:
        import torch._dynamo as dynamo
        print("dynamo       :", dynamo.__version__ if hasattr(dynamo, "__version__") else "有")
    except Exception as exc:
        print("dynamo       : 不可用", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
