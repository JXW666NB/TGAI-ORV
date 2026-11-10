# -*- coding: utf-8 -*-
"""查 C 端是否有 fp16 GEMM 能力（cuBLAS HGEMM / gemmEx），这是实现 AMP 的前提。"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
k = io.open(os.path.join(ROOT, "python", "tgai_go", "train", "cuda.py"), encoding="utf-8").read()

pats = ["cuBLASGemmEx", "GemmEx", "CUDA_R_16F", "CUDA_R_16BF", "HGEMM", "hgemm",
        "sgemm_mode", "def gemm", "CUBLAS_COMPUTE_32F", "CUBLAS_COMPUTE_16F"]
for p in pats:
    n = k.count(p)
    print("%-22s %d" % (p, n))

print()
i = k.find("def sgemm_mode")
if i > 0:
    print("=== sgemm_mode 附近 ===")
    print(k[i:i + 600])
