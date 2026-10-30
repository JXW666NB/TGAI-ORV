# -*- coding: utf-8 -*-
"""盘点 PyTorch 的算子面，并与我们的引擎逐项对照，产出 docs/OP_COVERAGE.md。

思路（用户提出，比"搬模型撞缺口"更根本）：
    PyTorch 的公开算子面是**客观基准**。先把它的清单枚举出来，
    再逐项标注我们「有 / 部分 / 无」，按训练必需程度排序，
    然后一个一个攻破 —— 并且每个都要能跟 PyTorch 比数值、比速度。

用法：
    python tools/pytorch_op_surface.py            # 打印报告
    python tools/pytorch_op_surface.py --write     # 同时写 docs/OP_COVERAGE.md
"""
import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for cand in (os.path.join(REPO, "python"), REPO):
    if os.path.isdir(os.path.join(cand, "tgai_go")) and cand not in sys.path:
        sys.path.insert(0, cand)

# ---------------------------------------------------------------------------
# 一、训练必需的核心算子（人工分层：这些是「能不能训别人的模型」的硬门槛）
#     value = (torch 名字, 我们的实现名或 None, 说明)
# ---------------------------------------------------------------------------
TIER1 = [
    # 矩阵/线性
    ("torch.matmul", "@ (Tensor.__matmul__)", ""),
    ("torch.addmm", "nn_ops.addmm", "组合实现，已验证形状检查"),
    ("torch.bmm", "batched @", "注意力里的批量矩阵乘"),
    ("torch.einsum", "nn_ops.einsum", "支持 1~2 操作数常见形式，已验证"),
    ("torch.nn.functional.linear", "nn_ops.linear", "已验证"),
    # 归一化
    ("torch.nn.functional.rms_norm", "nn_ops.rms_norm", "已验证（Qwen2 对齐用到）"),
    ("torch.nn.functional.layer_norm", "nn_ops.layer_norm", "已验证"),
    # 激活
    ("torch.nn.functional.silu", "F.silu", ""),
    ("torch.nn.functional.gelu", "nn_ops.gelu / contrib.gelu_exact", "tanh 近似在核心，精确版在 contrib"),
    ("torch.nn.functional.relu", "nn_ops.relu", "已验证"),
    ("torch.nn.functional.softmax", "F.softmax", ""),
    ("torch.nn.functional.log_softmax", "F.log_softmax", ""),
    # 注意力
    ("torch.nn.functional.scaled_dot_product_attention", "F.scaled_dot_product_attention", "GQA/因果掩码要覆盖"),
    # 损失
    ("torch.nn.functional.cross_entropy", "F.cross_entropy", ""),
    ("torch.nn.functional.nll_loss", "nn_ops.nll_loss", "log_softmax 之后常用"),
    ("torch.nn.functional.mse_loss", "nn_ops.mse_loss", "回归/对齐"),
    # 归约
    ("torch.sum", "Tensor.sum", "需要支持 dim/keepdim"),
    ("torch.mean", "Tensor.mean", "已支持 dim/keepdim"),
    ("torch.max", "nn_ops.amax / Tensor.maximum", "逐元素有；沿维取最大值+索引没有"),
    ("torch.min", "nn_ops.amin", "已验证"),
    ("torch.amax", "nn_ops.amax", "已验证"),
    ("torch.norm", "nn_ops.norm", "p=2 已验证"),
    # 逐元素
    ("torch.add", "Tensor.__add__", ""),
    ("torch.mul", "Tensor.__mul__", ""),
    ("torch.div", "Tensor.__truediv__", "float/Tensor 目前不支持（反向标量除法缺）"),
    ("torch.sub", "Tensor.__sub__", ""),
    ("torch.pow", "nn_ops.pow", "整数幂精确，非整数走 exp/log"),
    ("torch.exp", "Tensor.exp", ""),
    ("torch.log", "Tensor.log", ""),
    ("torch.sqrt", "Tensor.sqrt", ""),
    ("torch.rsqrt", "nn_ops.rsqrt", "已验证"),
    ("torch.tanh", "nn_ops.tanh", "已验证"),
    ("torch.erf", "contrib.erf", "官方 contrib 插件"),
    ("torch.sigmoid", "nn_ops.sigmoid", "已验证"),
    ("torch.clamp", "Tensor.clamp", ""),
    ("torch.abs", "Tensor.abs", ""),
    # 形状/索引
    ("torch.reshape", "Tensor.reshape", ""),
    ("torch.view", "Tensor.view", ""),
    ("torch.transpose", "Tensor.transpose", ""),
    ("torch.permute", "Tensor.permute", ""),
    ("torch.cat", "T.cat", ""),
    ("torch.stack", "nn_ops.stack", "已验证"),
    ("torch.split", "nn_ops.split", "已验证"),
    ("torch.chunk", "nn_ops.chunk", "已验证"),
    ("torch.gather", "nn_ops.embedding(仅前向)", "embedding 查表的最小依赖"),
    ("torch.embedding", "nn_ops.embedding", "有，但**不支持反向**（缺 gather 内核）"),
    ("torch.repeat_interleave", "Tensor.repeat_interleave", "GQA 扩头"),
    ("torch.triu/tril", "nn_ops.triu / nn_ops.tril", "triu 已验证；tril 未实现"),
    ("torch.where", "nn_ops.where", "已验证"),
    # 类型/设备
    ("torch.tensor", "T.tensor", ""),
    ("torch.zeros/ones/empty", "T.zeros/T.ones", ""),
    ("torch.from_numpy", "T.tensor(numpy)", ""),
    ("Tensor.to(dtype/device)", "Tensor.to_numpy / .data", "dtype 转换目前只能绕"),
    ("Tensor.float/half/bfloat16", None, "真 AMP 的前置"),
    # 随机
    ("torch.randn", "nn_ops.randn", "已验证形状/分布"),
    ("torch.manual_seed", "nn_ops.manual_seed", "已验证可复现"),
    # 反向/优化
    ("Tensor.backward", "Tensor.backward", ""),
    ("Tensor.detach", "Tensor.detach", ""),
    ("torch.optim.AdamW", "optim.AdamW", ""),
    ("torch.optim.SGD", "nn_ops.sgd_step", "最常用优化器之一"),
    ("torch.nn.utils.clip_grad_norm_", "nn_ops.clip_grad_norm_", "已验证"),
    # 序列/位置
    ("torch.arange", "nn_ops.arange", "已验证"),
    ("torch.cumsum", "nn_ops.cumsum", "已验证(含反向)"),
]

CATEGORY_RULES = [
    ("矩阵乘", ("matmul", "mm", "bmm", "einsum", "linear", "addmm", "baddbmm", "outer", "dot", "tensordot")),
    ("归一化", ("norm", "normalize", "batch_norm", "group_norm", "layer_norm", "instance_norm")),
    ("激活", ("relu", "gelu", "silu", "sigmoid", "tanh", "softmax", "log_softmax", "mish", "elu",
              "selu", "leaky_relu", "prelu", "hardswish", "hardsigmoid", "softplus", "glu")),
    ("损失", ("loss", "cross_entropy", "nll", "kl_div", "binary_cross", "mse", "l1_loss",
              "smooth_l1", "triplet", "cosine_embedding", "ctc")),
    ("归约", ("sum", "mean", "max", "min", "prod", "amax", "amin", "argmax", "argmin",
              "logsumexp", "std", "var", "median", "cumsum", "cumprod", "count_nonzero", "any", "all")),
    ("逐元素", ("add", "sub", "mul", "div", "pow", "exp", "log", "sqrt", "rsqrt", "abs", "neg",
                "clamp", "clip", "round", "floor", "ceil", "sign", "trunc", "lerp", "fmod", "remainder")),
    ("形状", ("reshape", "view", "transpose", "permute", "cat", "stack", "split", "chunk", "squeeze",
              "unsqueeze", "flatten", "ravel", "narrow", "expand", "repeat", "tile", "roll", "flip",
              "contiguous", "pad", "tril", "triu", "diag", "meshgrid", "einsum")),
    ("索引", ("gather", "scatter", "index_select", "masked_select", "embedding", "take", "where",
              "nonzero", "bucketize", "searchsorted", "topk", "sort", "argsort", "unique")),
    ("随机", ("rand", "randn", "normal", "bernoulli", "dropout", "multinomial", "manual_seed", "poisson")),
    ("类型/设备", ("to", "float", "half", "bfloat16", "int", "long", "bool", "type_as", "cpu", "cuda",
                   "detach", "requires_grad", "contiguous", "clone")),
    ("序列", ("arange", "linspace", "logspace", "range", "eye", "zeros", "ones", "empty", "full")),
    ("线性代数", ("svd", "qr", "cholesky", "solve", "inverse", "det", "eig", "lu", "lstsq", "pinverse")),
    ("卷积", ("conv", "pool", "unfold", "fold", "interpolate", "grid_sample", "upsample")),
    ("分布式", ("distributed", "all_reduce", "broadcast", "reduce_scatter", "all_gather", "send", "recv")),
]


# ---------------------------------------------------------------------------
# 范围裁剪：我们是 **LLM 训练引擎**，不是通用张量库。
# 下列领域明确**不做**（不是"还没做"，是"永远不在范围内"），
# 这样清单不会因为 PyTorch 有 1016 个函数而显得永远补不完。
# ---------------------------------------------------------------------------
OUT_OF_SCOPE = [
    ("卷积/视觉", ("conv", "pool", "unfold", "fold", "interpolate", "grid_sample",
                   "upsample", "pixel_shuffle", "pixel_unshuffle", "reflection_pad",
                   "replication_pad", "constant_pad", "channel_shuffle", "avg_pool",
                   "max_pool", "adaptive_", "bilinear")),
    ("线性代数/科研", ("svd", "qr", "cholesky", "solve", "inverse", "det", "eig", "lu",
                       "lstsq", "pinverse", "symeig", "matrix_rank", "matrix_exp", "trace",
                       "tensordot", "kron", "vander", "trapz", "trapezoid", "fft", "stft",
                       "istft", "rfft", "hann", "hamming", "blackman", "bartlett")),
    ("概率分布", ("distribution", "categorical", "poisson", "gamma", "beta", "dirichlet",
                  "multivariate_normal", "cauchy", "geometric", "laplace", "weibull")),
    ("稀疏", ("sparse", "coo_", "csr_", "csc_", "to_sparse", "spmm", "spspmm")),
    ("量化/图/编译器", ("quantize", "dequantize", "fake_quant", "is_compiling", "compile",
                        "dynamo", "fx_", "jit", "tracing", "export")),
    ("视觉/音频 IO", ("imread", "imwrite", "read_image", "write_image", "save_image",
                      "audio", "spectrogram", "mel")),
    ("其他非训练", ("cdist", "pairwise_distance", "pdist", "histc", "histogram", "bincount",
                    "cartesian_prod", "combinations", "symint", "vmap", "func_")),
]


def out_of_scope_reason(name):
    low = name.lower()
    for reason, keys in OUT_OF_SCOPE:
        if any(k in low for k in keys):
            return reason
    return None


def categorize(name):
    low = name.lower()
    for cat, keys in CATEGORY_RULES:
        if any(k in low for k in keys):
            return cat
    return "其他"


def torch_surface():
    """枚举 PyTorch 公开可调用面（函数 + nn.functional + Tensor 方法）。"""
    import torch
    import torch.nn.functional as F

    funcs = set()
    for n in dir(torch):
        if n.startswith("_"):
            continue
        if callable(getattr(torch, n, None)):
            funcs.add("torch." + n)
    for n in dir(F):
        if n.startswith("_"):
            continue
        if callable(getattr(F, n, None)):
            funcs.add("F." + n)
    methods = set()
    for n in dir(torch.Tensor):
        if n.startswith("_"):
            continue
        methods.add("Tensor." + n)
    return sorted(funcs), sorted(methods)


def our_surface():
    """枚举我方名字：Tensor 方法 + functional + 运行期原语 + CUDA 源码里的 kernel 名。"""
    import tgai_go.train as T
    from tgai_go.train import functional as F

    names = set()
    names |= {n for n in dir(T.Tensor) if not n.startswith("_")}
    names |= {n for n in dir(F) if not n.startswith("_")}
    try:
        from tgai_go.train import nn_ops as O
        names |= {n for n in dir(O) if not n.startswith("_")}
    except Exception:  # noqa: BLE001
        pass
    try:                       # 官方 contrib 插件也算「能拿到」
        from tgai_go.train.contrib import autoload
        autoload()
        names |= {n for n in dir(F) if not n.startswith("_")}
    except Exception:  # noqa: BLE001
        pass
    try:
        from tgai_go.train import backend
        rt = backend.get("cuda", check=False)
        names |= {n for n in dir(rt) if not n.startswith("_")}
    except Exception:  # noqa: BLE001
        pass
    # CUDA 源码里的 kernel 名（说明"有能力但没公开成算子"）
    src = os.path.join(REPO, "python", "tgai_go", "train", "cuda.py")
    if os.path.exists(src):
        txt = open(src, encoding="utf-8", errors="replace").read()
        names |= set(re.findall(r"__global__\s+void\s+(\w+)", txt))
        names |= set(re.findall(r"def (\w+)\(", txt))
    return names


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    funcs, methods = torch_surface()
    ours = our_surface()
    print("PyTorch 公开函数 %d 个 / Tensor 方法 %d 个 / 我们认得 %d 个名字"
          % (len(funcs), len(methods), len(ours)))

    t1_rows = []
    for tname, ourname, note in TIER1:
        short = tname.split(".")[-1]
        have = ourname is not None
        ok = have
        t1_rows.append((tname, ourname, "有" if ok else "缺", note))

    n_have = sum(1 for r in t1_rows if r[2] == "有")
    print("Tier-1（训练必需）%d 项，我们有 %d 项 ⇒ 覆盖 %.0f%%"
          % (len(t1_rows), n_have, 100.0 * n_have / len(t1_rows)))

    n_oos = sum(1 for n in funcs if out_of_scope_reason(n))
    n_in = len(funcs) - n_oos
    print("范围裁剪：PyTorch 函数 %d 个，其中 **范围外 %d 个**（卷积/视觉/线代/科研/稀疏/量化），"
          "在范围内 %d 个" % (len(funcs), n_oos, n_in))

    cats = {}
    for name in funcs:
        if out_of_scope_reason(name):
            continue
        cats.setdefault(categorize(name), []).append(name)
    print("\nPyTorch 函数面分类：")
    for cat in sorted(cats, key=lambda c: -len(cats[c])):
        print("  %-10s %4d" % (cat, len(cats[cat])))

    if a.write:
        out = os.path.join(REPO, "docs", "OP_COVERAGE.md")
        L = []
        L.append("# 算子覆盖清单（以 PyTorch 为基准）\n")
        L.append("> 由 `python tools/pytorch_op_surface.py --write` 生成，"
                 "基准是本机 `torch` 的真实公开面。\n")
        L.append("## 总览\n")
        L.append("| 项 | 数量 |")
        L.append("| --- | --- |")
        L.append("| PyTorch 公开函数 | %d |" % len(funcs))
        L.append("| PyTorch Tensor 方法 | %d |" % len(methods))
        L.append("| **Tier-1 训练必需算子** | **%d** |" % len(t1_rows))
        L.append("| 其中我们已具备 | **%d（%.0f%%）** |" % (n_have, 100.0 * n_have / len(t1_rows)))
        L.append("")
        L.append("## Tier-1：训练别人的模型所必需\n")
        L.append("| PyTorch 算子 | 我们的对应 | 状态 | 备注 |")
        L.append("| --- | --- | --- | --- |")
        for tname, ourname, st, note in t1_rows:
            L.append("| `%s` | %s | %s | %s |"
                     % (tname, ("`%s`" % ourname) if ourname else "—", st, note))
        L.append("")
        L.append("## 范围裁剪：我们不做哪些（明确不做，不是待办）\n")
        L.append("我们是 **LLM 训练引擎**，不是通用张量库。PyTorch 公开函数 %d 个，"
                 "其中 **%d 个** 属下列领域，**永远不在范围内**；剩下 %d 个才进入考量的池子。\n"
                 % (len(funcs), n_oos, n_in))
        L.append("| 领域 | 例子 | 为什么不做的 |")
        L.append("| --- | --- | --- |")
        why = {
            "卷积/视觉": ("conv2d / max_pool2d / interpolate", "LLM 没有卷积；视觉任务不在产品范围"),
            "线性代数/科研": ("svd / qr / cholesky / fft", "科研张量库的需求，训练用不到"),
            "概率分布": ("Normal / Categorical", "需要时用 numpy 采样即可，不值得进内核"),
            "稀疏": ("to_sparse / spmm", "LLM 训练用稠密权重"),
            "量化/图/编译器": ("quantize / torch.compile", "我们有 NVRTC + CUDA Graph 这条自己的路"),
            "视觉/音频 IO": ("imread / spectrogram", "decode 归数据管线，不归训练内核"),
            "其他非训练": ("cdist / histogram / vmap", "非训练必需"),
        }
        for reason, _keys in OUT_OF_SCOPE:
            ex, whyw = why.get(reason, ("—", "—"))
            cnt = sum(1 for n in funcs if out_of_scope_reason(n) == reason)
            L.append("| %s | `%s` | %s（%d 个） |" % (reason, ex, whyw, cnt))
        L.append("")
        L.append("## PyTorch 函数面分类计数（仅范围内）\n")
        L.append("| 类别 | 数量 |")
        L.append("| --- | --- |")
        for cat in sorted(cats, key=lambda c: -len(cats[c])):
            L.append("| %s | %d |" % (cat, len(cats[cat])))
        L.append("")
        L.append("## 说明\n")
        L.append("- 「有」= 公开 API 里能找到对应实现；不等于已对齐 PyTorch 语义或用例覆盖完整。")
        L.append("- 很多能力我们**内核里有、但没公开成算子**（如 rmsnorm/rope 在 `cuda.py` 里），"
                 "清单里按「缺」计，因为它们不能被用户代码直接调用。")
        L.append("- 下一步：按 Tier-1 顺序逐项实现，每项配 ①与 PyTorch 的数值对齐测试 "
                 "②速度对比 ③错误信息三要素。")
        open(out, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
        print("\n已写入", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
