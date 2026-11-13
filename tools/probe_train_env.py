# -*- coding: utf-8 -*-
"""环境与可行性探测：参数总量、PyTorch 是否可用、显存、语料规模。

只读操作，不启动训练、不碰产线推理进程。

用法：`python probe_train_env.py <model.tg>`
"""
import ctypes
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "python"))


def gpu_memory():
    """用 nvidia-smi 拿总/已用/空闲显存（MiB）。失败返回 None。"""
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        out = (r.stdout or "").strip().splitlines()
        if not out:
            return None
        name, total, used, free = [x.strip() for x in out[0].split(",")[:4]]
        return {"name": name, "total_mib": int(total), "used_mib": int(used),
                "free_mib": int(free)}
    except Exception as exc:                              # noqa: BLE001
        return {"error": "%s: %s" % (type(exc).__name__, exc)}


def torch_info():
    try:
        import torch
    except Exception as exc:                              # noqa: BLE001
        return {"available": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:120])}
    d = {"available": True, "version": torch.__version__,
         "cuda_build": torch.version.cuda,
         "cuda_available": bool(torch.cuda.is_available())}
    if d["cuda_available"]:
        d["device_name"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        d["free_mib"] = int(free // 1024 // 1024)
        d["total_mib"] = int(total // 1024 // 1024)
    return d


def param_stats(path):
    """从 .TG 的权重表算参数总量与分布（按 dtype 分类）。"""
    from tgai_go.read_tg import read_tg, close_tg
    info = read_tg(path)
    try:
        h = info["header"]
        total = 0
        per_dtype = {}
        rows = []
        for e in info["table"]:
            n = 1
            for v in e["shape"]:
                n *= int(v)
            total += n
            per_dtype[e["dtype"]] = per_dtype.get(e["dtype"], 0) + n
            rows.append((n, e["dtype"], tuple(e["shape"])))
        rows.sort(reverse=True)
        return {"path": path, "bytes": os.path.getsize(path),
                "header": {k: h.get(k) for k in
                           ("n_layers", "d_model", "n_heads", "d_ff", "vocab",
                            "max_seq", "n_experts", "n_activated", "num_kv_heads",
                            "weight_dtype", "head_dim")},
                "n_tensors": len(info["table"]),
                "total_params": total,
                "total_params_b": round(total / 1e9, 3),
                "per_dtype": per_dtype,
                "largest": [{"elems": r[0], "dtype": r[1], "shape": list(r[2])}
                            for r in rows[:5]]}
    finally:
        close_tg(info)


def memory_budget(total_params):
    """按几种常见训练配置估算显存（GiB）。**租借/购买前先看这个数**。"""
    P = float(total_params)
    gib = 1024.0 ** 3
    out = {}
    # ① 全参 + fp32 主权重 + fp32 Adam
    out["full_fp32_master"] = {
        "weights_bf16": 2 * P / gib, "grads_fp32": 4 * P / gib,
        "adam_m_fp32": 4 * P / gib, "adam_v_fp32": 4 * P / gib}
    # ② 全参 + 纯 fp16（权重/梯度/动量都 fp16，我们 Fp16MomAdamW 的能力）
    out["full_fp16_state"] = {
        "weights_fp16": 2 * P / gib, "grads_fp16": 2 * P / gib,
        "adam_m_fp16": 2 * P / gib, "adam_v_fp16": 2 * P / gib}
    for k, parts in out.items():
        parts["subtotal"] = round(sum(parts.values()), 2)
    return out


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else ""
    print("=== GPU ===")
    print(json.dumps(gpu_memory(), ensure_ascii=False))
    print("=== PyTorch ===")
    print(json.dumps(torch_info(), ensure_ascii=False))
    if model:
        print("=== 模型 ===")
        st = param_stats(model)
        print(json.dumps({k: st[k] for k in ("header", "n_tensors", "total_params_b",
                                            "bytes", "largest")},
                         ensure_ascii=False))
        print("=== 显存估算（不含激活）===")
        print(json.dumps(memory_budget(st["total_params"]), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
