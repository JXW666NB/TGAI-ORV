# -*- coding: utf-8 -*-
"""权重存储精度：float32 / float16 / bfloat16。

为什么需要：PyTorch 的 `state_dict` 保留张量 dtype，safetensors 也原生支持 bf16；
存检查点时用低精度能把文件减半（bf16 与 fp32 同样指数范围，**不损失动态范围**），
读取方再按需 upcast。以前我们只有 float32，这一块是明确的功能差距。

bf16 没有 numpy 原生类型，这里用 **uint16 存高位截断**（round-to-nearest-even），
与 CUDA 的 `__float2bfloat16` 语义一致；读回时左移 16 位还原成 float32。
"""
import json
import os

import numpy as np

SUPPORTED = ("float32", "float16", "bfloat16")


def _norm_dtype(dtype):
    d = str(dtype).strip().lower().replace("torch.", "")
    alias = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16",
             "half": "float16", "float": "float32"}
    d = alias.get(d, d)
    if d not in SUPPORTED:
        raise ValueError(
            "不支持的存储精度 %r。可用：%s（修法：传 dtype='float16' 或 'bfloat16'）"
            % (dtype, ", ".join(SUPPORTED)))
    return d


def f32_to_bf16_bits(a):
    """float32 -> bf16 位模式（uint16，round-to-nearest-even）。"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    # NaN 保持 NaN
    nan = np.isnan(a)
    lsb = (u >> 16) & 1
    rounded = (u + 0x7FFF + lsb) >> 16
    out = rounded.astype(np.uint16)
    if nan.any():
        out = np.where(nan, np.uint16(0x7FC0), out)
    return out


def bf16_bits_to_f32(b):
    """bf16 位模式（uint16）-> float32。"""
    b = np.ascontiguousarray(b, dtype=np.uint16)
    return (b.astype(np.uint32) << 16).view(np.float32)


def to_storage(arr, dtype):
    """把 float32 权重转成存储精度。bf16 返回 uint16 位模式。"""
    d = _norm_dtype(dtype)
    a = np.asarray(arr, dtype=np.float32)
    if d == "float32":
        return a
    if d == "float16":
        return a.astype(np.float16)
    return f32_to_bf16_bits(a)


def from_storage(arr, dtype):
    """存储精度 -> float32。"""
    d = _norm_dtype(dtype)
    if d == "float32":
        return np.asarray(arr, dtype=np.float32)
    if d == "float16":
        return np.asarray(arr, dtype=np.float16).astype(np.float32)
    return bf16_bits_to_f32(arr)


def cast_weights(weights, dtype):
    """整份权重换成 float32 视图（用于把 fp16/bf16 检查点载进训练）。"""
    return {k: from_storage(v, dtype) for k, v in weights.items()}


def save_weights(path, weights, dtype="float32", config=None):
    """按 dtype 存权重（fp32/fp16/bf16）。每个张量的 dtype 记在 `__dtypes__` 里，
    所以同一个文件也允许混合精度（fp32 的 norm + bf16 的线性层）。"""
    d = _norm_dtype(dtype)
    payload = {}
    meta = {}
    for k, v in weights.items():
        a = np.asarray(v)
        if a.dtype in (np.float16, np.uint16):
            raise ValueError(
                "save_weights 只接受 float32 权重（%s 是 %s）。修法：先 from_storage() 还原"
                % (k, a.dtype))
        payload[k] = to_storage(a, d)
        meta[k] = d
    payload["__dtypes__"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    if config:
        payload["__config__"] = np.frombuffer(
            json.dumps(config, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
    np.savez(path, **payload)
    return path


def load_weights(path, upcast=True):
    """读回 save_weights 存的文件。upcast=True 时还原成 float32（训练用）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError("找不到检查点：%s。修法：检查路径" % (path,))
    z = np.load(path, allow_pickle=False)
    meta = {}
    if "__dtypes__" in z.files:
        try:
            meta = json.loads(bytes(z["__dtypes__"]).decode("utf-8"))
        except Exception:                          # noqa: BLE001
            meta = {}
    cfg = {}
    if "__config__" in z.files:
        try:
            cfg = json.loads(bytes(z["__config__"]).decode("utf-8"))
        except Exception:                          # noqa: BLE001
            cfg = {}
    out = {}
    for k in z.files:
        if k in ("__dtypes__", "__config__"):
            continue
        d = meta.get(k, "float32")
        out[k] = from_storage(z[k], d) if upcast else z[k]
    return out, cfg
