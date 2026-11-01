# -*- coding: utf-8 -*-
"""检查点读写：GGUF / safetensors / npz 统一入口。

设计目标是回答一个问题：**「别人的模型」能不能直接拿来训练？**

大多数中文模型的发布形态是 GGUF。所以这里把 GGUF 当成**一等公民的检查点格式**，
不需要先转成 .TG 或 .pt：

    from tgai_go.train import checkpoints as C
    ck = C.load_gguf(r"D:\\models\\qwen2.5-0.5b-fp16.gguf")
    ck.weights["token_embedding.weight"]     # np.float32
    ck.config["n_layer"], ck.config["rope_theta"]

三条原则（对应踩过的坑）：
1. **不认识的架构/量化类型要明确报错并给修法**，不许静默丢权重
   （Qwen3 的 `attn_q_norm/attn_k_norm` 就是典型：早期转换器把它们丢了，
   模型能加载但输出全错，且没有任何提示）。
2. **名字映射只有一处**（本模块），工具与示例都从这儿取，避免两套映射漂移。
3. 缺失必需权重时列出**缺哪几个**，而不是等训练时才崩。
"""
import os

import numpy as np

#: 支持的 GGUF 架构（GGUF 里的 `general.architecture`）
SUPPORTED_ARCHES = ("llama", "mistral", "qwen2", "qwen3")

#: GGUF 张量名后缀 → 我们的命名（layers 之外的部分）
GLOBAL_SUFFIX = {
    "token_embd.weight": "token_embedding.weight",
    "output_norm.weight": "final_norm.weight",
    "output.weight": "lm_head.weight",
}

#: 每层的张量名后缀 → 我们的命名
LAYER_SUFFIX = {
    "attn_norm.weight": "ln1.weight",
    "ffn_norm.weight": "ln2.weight",
    "attn_q.weight": "attn.q_proj.weight",
    "attn_k.weight": "attn.k_proj.weight",
    "attn_v.weight": "attn.v_proj.weight",
    "attn_output.weight": "attn.out_proj.weight",
    "attn_q.bias": "attn.q_proj.bias",
    "attn_k.bias": "attn.k_proj.bias",
    "attn_v.bias": "attn.v_proj.bias",
    "attn_output.bias": "attn.out_proj.bias",
    "ffn_gate.weight": "ffn.w1.weight",
    "ffn_up.weight": "ffn.w2.weight",
    "ffn_down.weight": "ffn.w3.weight",
    # Qwen3：每个 head 上的 RMSNorm（少了它输出就是错的）
    "attn_q_norm.weight": "attn.q_norm.weight",
    "attn_k_norm.weight": "attn.k_norm.weight",
}

#: 各架构「必须有」的每层张量（缺了就直接报错，不等训练时才发现）
REQUIRED_LAYER_KEYS = {
    "llama": ("attn_norm", "ffn_norm", "attn_q", "attn_k", "attn_v", "attn_output",
              "ffn_gate", "ffn_up", "ffn_down"),
    "mistral": ("attn_norm", "ffn_norm", "attn_q", "attn_k", "attn_v", "attn_output",
                "ffn_gate", "ffn_up", "ffn_down"),
    "qwen2": ("attn_norm", "ffn_norm", "attn_q", "attn_k", "attn_v", "attn_output",
              "ffn_gate", "ffn_up", "ffn_down"),
    "qwen3": ("attn_norm", "ffn_norm", "attn_q", "attn_k", "attn_v", "attn_output",
              "ffn_gate", "ffn_up", "ffn_down", "attn_q_norm", "attn_k_norm"),
}

#: 支持直接读取的 GGML 浮点类型（量化类型需要先反量化，见错误提示）
_FLOAT_TYPES = ("F32", "F16", "BF16")

#: GGUF 里读的架构超参 → 我们的 cfg 键
_META_KEYS = {
    "block_count": "n_layer",
    "embedding_length": "n_embd",
    "attention.head_count": "n_head",
    "attention.head_count_kv": "n_head_kv",
    "feed_forward_length": "n_ff",
    "attention.layer_norm_rms_epsilon": "eps",
    "rope.freq_base": "rope_theta",
    "attention.key_length": "head_dim",
}



def _as_dict(obj):
    """gguf 的不同版本里 fields/tensors 可能是 dict 或 list，统一成 {名字: 对象}。"""
    if isinstance(obj, dict):
        return obj
    out = {}
    for item in obj:
        name = getattr(item, "name", None)
        if name is not None:
            out[name] = item
    return out


def _meta_value(node):
    """从 ReaderField 取标量值（不同版本 API 略有差异）。"""
    if node is None:
        return None
    for attr in ("contents", "value"):
        if hasattr(node, attr):
            try:
                v = getattr(node, attr)() if attr == "contents" else getattr(node, attr)
            except Exception:                      # noqa: BLE001
                continue
            if isinstance(v, (list, tuple)):
                v = v[0] if len(v) == 1 else v
            if isinstance(v, bytes):
                try:
                    return v.decode("utf-8")
                except UnicodeDecodeError:
                    return v
            return v
    return None


class CheckpointError(ValueError):
    """检查点读取错误。消息里必须写清「哪一步 / 什么问题 / 怎么修」。"""


class Checkpoint:
    """一个可训练检查点：权重（np.float32）+ 结构配置。"""

    def __init__(self, weights, config, source=""):
        self.weights = weights
        self.config = config
        self.source = source

    def __len__(self):
        return len(self.weights)

    def __repr__(self):
        c = self.config
        return ("<Checkpoint %s arch=%s L=%s D=%s NH=%s/KV=%s FF=%s V=%s tie=%s>"
                % (os.path.basename(self.source), c.get("arch"), c.get("n_layer"),
                   c.get("n_embd"), c.get("n_head"), c.get("n_head_kv"),
                   c.get("n_ff"), c.get("vocab_size"), c.get("tie_embeddings")))

    @property
    def tied(self):
        return bool(self.config.get("tie_embeddings", False))


def map_gguf_name(gguf_name, arch="qwen2", n_layer=None):
    """GGUF 张量名 → 我们的命名。不认识就返回 None（由调用方决定是否报错）。"""
    if gguf_name in GLOBAL_SUFFIX:
        return GLOBAL_SUFFIX[gguf_name]
    if not gguf_name.startswith("blk."):
        return None
    parts = gguf_name.split(".", 2)
    if len(parts) < 3:
        return None
    try:
        layer = int(parts[1])
    except ValueError:
        return None
    ours = LAYER_SUFFIX.get(parts[2])
    if ours is None:
        return None
    return "blocks.%d.%s" % (layer, ours)


def _tensor_to_numpy(t, name):
    """把一个 GGUF 张量转成 np.float32，类型不支持时给明确修法。"""
    ttype = getattr(t, "tensor_type", None)
    tname = getattr(ttype, "name", str(ttype))
    if tname not in _FLOAT_TYPES:
        raise CheckpointError(
            "张量 %r 是量化类型 %s，训练需要浮点权重。"
            "修法：用 fp16/fp32 的 GGUF（如 *-fp16.gguf），或先用 llama.cpp 的 "
            "`llama-quantize ... F16` 转成 F16 再来。" % (name, tname))
    try:
        arr = np.asarray(t.data)
    except Exception as exc:                      # noqa: BLE001
        raise CheckpointError(
            "读取张量 %r（%s）失败：%s。修法：确认 GGUF 未损坏、版本受支持"
            % (name, tname, exc)) from exc
    if arr.dtype == np.uint16:                    # BF16 以 uint16 承载
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _read_metadata(reader):
    """从 GGUF 元数据读结构超参。缺关键项时报错，不猜。"""
    fields = _as_dict(reader.fields)
    tensors = _as_dict(reader.tensors)
    arch = _meta_value(fields.get("general.architecture"))
    arch = str(arch) if arch else "llama"
    if arch not in SUPPORTED_ARCHES:
        raise CheckpointError(
            "不支持的 GGUF 架构 %r。目前已支持：%s。"
            "修法：在 checkpoints.SUPPORTED_ARCHES 里补一项，并按 "
            "LAYER_SUFFIX 补张量名映射" % (arch, ", ".join(SUPPORTED_ARCHES)))
    cfg = {"arch": arch}
    for meta_key, ours in _META_KEYS.items():
        full = "%s.%s" % (arch, meta_key)
        val = _meta_value(fields.get(full))
        if val is None:
            continue
        cfg[ours] = val
    # 词表：优先 token_embd 的形状（比任何元数据都可靠）
    emb = tensors.get("token_embd.weight")
    if emb is None:
        raise CheckpointError(
            "GGUF 里没有 token_embd.weight。修法：确认这是完整的模型文件，"
            "不是只含部分张量的适配器（LoRA）")
    # 注意：GGUF 存储的 dims 是**反序**的（第一维在内层），ReaderTensor.shape 给的是
    # 存储顺序，直接用会把 vocab 和 hidden 弄反。用 .data 的实际 numpy 形状最可靠。
    try:
        emb_shape = tuple(np.asarray(emb.data).shape)
    except Exception:                              # noqa: BLE001
        emb_shape = tuple(reversed(tuple(int(x) for x in emb.shape)))
    cfg["vocab_size"] = int(emb_shape[0])
    if "n_embd" not in cfg:
        cfg["n_embd"] = int(emb_shape[1])
    if "n_head_kv" not in cfg:
        cfg["n_head_kv"] = cfg.get("n_head", 0)
    # 是否绑定输出投影：直接比对数据最可靠（GGUF 常常没有 tie 元数据）
    out = tensors.get("output.weight")
    tie = True
    if out is not None:
        try:
            oa = np.asarray(out.data).reshape(-1)
            ea = np.asarray(emb.data).reshape(-1)
            tie = bool(oa.size == ea.size
                       and np.array_equal(oa[:4096], ea[:4096])
                       and np.array_equal(oa[-4096:], ea[-4096:]))
        except Exception:                          # noqa: BLE001
            tie = True
    cfg["tie_embeddings"] = tie
    cfg.setdefault("rope_theta", 10000.0)
    cfg.setdefault("eps", 1e-6)
    return cfg


def load_gguf(path):
    """读一个 GGUF 文件 → Checkpoint（权重 + 结构配置）。

    只有 F32/F16/BF16 能直接读；量化类型会明确报错（见 CheckpointError 的修法）。
    """
    if not os.path.isfile(path):
        raise CheckpointError(
            "找不到 GGUF 文件：%s。修法：检查路径，或先用 modelscope/hf 下载"
            % (path,))
    try:
        import gguf
    except ImportError as exc:
        raise CheckpointError(
            "需要 gguf 包才能读 GGUF。修法：pip install gguf") from exc

    reader = gguf.GGUFReader(path)
    cfg = _read_metadata(reader)
    arch = cfg["arch"]

    tensors = _as_dict(reader.tensors)
    weights, dropped = {}, []
    for name, tensor in tensors.items():
        ours = map_gguf_name(name, arch)
        if ours is None:
            dropped.append(name)
            continue
        weights[ours] = _tensor_to_numpy(tensor, name)

    # 缺关键权重 → 列出具体名字，而不是等训练时崩
    missing = []
    if "token_embedding.weight" not in weights:
        missing.append("token_embd.weight")
    if "final_norm.weight" not in weights:
        missing.append("output_norm.weight")
    n_layer = int(cfg.get("n_layer") or 0)
    need = REQUIRED_LAYER_KEYS.get(arch, ())
    for i in range(n_layer):
        for key in need:
            ours = "blocks.%d.%s" % (i, LAYER_SUFFIX.get(key + ".weight", "").strip("."))
            ours = "blocks.%d.%s" % (i, LAYER_SUFFIX[key + ".weight"])
            if ours not in weights:
                missing.append("blk.%d.%s.weight" % (i, key))
                if len(missing) > 8:
                    break
        if len(missing) > 8:
            break
    if missing:
        raise CheckpointError(
            "GGUF 缺少必需权重（前几个）：%s。修法：确认文件完整；"
            "若这是 LoRA/部分权重，请先用基座模型合并" % (", ".join(missing[:8]),))

    # tie 时丢掉 output.weight：Qwen2.5 的 GGUF 里 output 与 token_embd 逐位相同，
    # 留着会被当成独立 lm_head，等于悄悄把 tie 模型训成不 tie。
    if cfg.get("tie_embeddings", True) and "lm_head.weight" in weights:
        weights.pop("lm_head.weight", None)

    # 没映射上的张量：可能是新架构的新增权重 —— 必须说出来，不能静默丢
    if dropped:
        cfg["dropped_tensors"] = dropped
        print("  注意：%d 个 GGUF 张量未映射（前 3 个：%s）"
              % (len(dropped), dropped[:3]))
    return Checkpoint(weights, cfg, source=path)


# ---------------------------------------------------------------------------
# safetensors（HuggingFace 官方目录）
# ---------------------------------------------------------------------------

_HF_PATTERNS = [
    (r"^model\.embed_tokens\.weight$", "token_embedding.weight"),
    (r"^lm_head\.weight$", "lm_head.weight"),
    (r"^model\.norm\.weight$", "final_norm.weight"),
    (r"^model\.layers\.(\d+)\.input_layernorm\.weight$", r"blocks.\1.ln1.weight"),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$", r"blocks.\1.ln2.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$", r"blocks.\1.attn.q_proj.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$", r"blocks.\1.attn.k_proj.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$", r"blocks.\1.attn.v_proj.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$", r"blocks.\1.attn.out_proj.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_norm\.weight$", r"blocks.\1.attn.q_norm.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_norm\.weight$", r"blocks.\1.attn.k_norm.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$", r"blocks.\1.ffn.w1.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$", r"blocks.\1.ffn.w2.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$", r"blocks.\1.ffn.w3.weight"),
]


def hf_name_to_ours(name):
    """HuggingFace 参数名 → 我们的命名（不认识返回 None）。"""
    import re
    for pat, rep in _HF_PATTERNS:
        if re.match(pat, name):
            return re.sub(pat, rep, name)
    return None


def load_safetensors_dir(dirpath):
    """读 HF 权重目录（model.safetensors）→ Checkpoint。

    bf16 权重 numpy 读不了，会自动走 torch 转换；两者都没有时明确报错。
    """
    path = os.path.join(dirpath, "model.safetensors")
    if not os.path.isfile(path):
        raise CheckpointError(
            "找不到 %s。修法：确认目录里有 model.safetensors（HF 官方目录）" % (path,))
    try:
        from safetensors.numpy import load_file
        raw = load_file(path)
    except TypeError as exc:
        if "bfloat16" not in str(exc):
            raise
        try:
            from safetensors.torch import load_file as tload
        except ImportError as exc2:
            raise CheckpointError(
                "权重是 bfloat16，numpy 读不了。修法：pip install torch（用它转 fp32），"
                "或先把权重转成 fp16/fp32") from exc2
        raw = {k: v.float().numpy() for k, v in tload(path).items()}

    weights, dropped = {}, []
    for k, v in raw.items():
        ours = hf_name_to_ours(k)
        if ours is None:
            dropped.append(k)
            continue
        weights[ours] = np.ascontiguousarray(np.asarray(v, dtype=np.float32))
    if "token_embedding.weight" not in weights:
        raise CheckpointError(
            "缺少 model.embed_tokens.weight。修法：确认是 Qwen2/Qwen3/LLaMA 结构")
    # ★ 结构参数**必须**读 config.json：Qwen3-0.6B 的 head_dim=128 而 hidden/heads=64，
    #   只靠权重形状猜会退化、直接算错（reshape 就崩）。tie 也要以 config 为准：
    #   官方 safetensors 里**带 lm_head.weight**，用"有没有"判 tie 会把 tie 模型训成不 tie。
    cfg = {"arch": "qwen3" if "blocks.0.attn.q_norm.weight" in weights else "qwen2"}
    jpath = os.path.join(dirpath, "config.json")
    hf = {}
    if os.path.isfile(jpath):
        import json
        with open(jpath, encoding="utf-8") as f:
            hf = json.load(f)
    if hf:
        D = int(hf["hidden_size"])
        NH = int(hf["num_attention_heads"])
        cfg.update({
            "n_layer": int(hf["num_hidden_layers"]),
            "n_embd": D,
            "n_head": NH,
            "n_head_kv": int(hf.get("num_key_value_heads", NH)),
            "head_dim": int(hf.get("head_dim") or D // NH),
            "n_ff": int(hf.get("intermediate_size") or 0),
            "vocab_size": int(hf.get("vocab_size") or 0),
            "rope_theta": float(hf.get("rope_theta", 10000.0)),
            # rope_scaling 必须读：Llama-3 / Mistral-v0.3 等靠它把频率分段缩放。
            # 不读 = 静默忽略 = 长上下文算错且不报错。
            "rope_scaling": hf.get("rope_scaling") or None,
            "eps": float(hf.get("rms_norm_eps", 1e-6)),
            "tie_embeddings": bool(hf.get("tie_word_embeddings", True)),
        })
    else:
        cfg["tie_embeddings"] = "lm_head.weight" not in weights
    n_layer = cfg.get("n_layer")
    if n_layer is None:
        n_layer = max((int(k.split(".")[1]) for k in weights if k.startswith("blocks.")),
                      default=-1) + 1
        cfg["n_layer"] = n_layer
    emb = weights["token_embedding.weight"]
    cfg.setdefault("vocab_size", int(emb.shape[0]) or 0)
    if not cfg.get("vocab_size"):
        cfg["vocab_size"] = int(emb.shape[0])
    cfg.setdefault("n_embd", int(emb.shape[1]))

    # tie 时丢掉重复的 lm_head（先校验相等，避免把"其实不 tie"的模型强行绑定）
    if cfg["tie_embeddings"] and "lm_head.weight" in weights:
        if not np.array_equal(weights["lm_head.weight"], weights["token_embedding.weight"]):
            raise CheckpointError(
                "config.json 说 tie_word_embeddings=true，但 lm_head 与词嵌入不一致。"
                "修法：确认权重目录完整（可能混入了别的模型的 lm_head）")
        weights.pop("lm_head.weight")
    if dropped:
        print("  注意：%d 个 HF 参数未映射（前 3 个：%s）" % (len(dropped), dropped[:3]))
    return Checkpoint(weights, cfg, source=dirpath)


# ---------------------------------------------------------------------------
# npz（训练后保存 / 续训）
# ---------------------------------------------------------------------------

def save_npz(weights, path, config=None):
    """把权重（np.float32）存成 npz。config 用 JSON 存进 `__config__`。"""
    import json
    payload = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
    if config:
        payload["__config__"] = np.frombuffer(
            json.dumps(config, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
    np.savez(path, **payload)
    return path


def load_npz(path):
    """读回 save_npz 存的权重 → Checkpoint。"""
    import json
    if not os.path.isfile(path):
        raise CheckpointError("找不到检查点：%s。修法：检查路径" % (path,))
    z = np.load(path, allow_pickle=False)
    cfg = {}
    weights = {}
    for k in z.files:
        if k == "__config__":
            try:
                cfg = json.loads(bytes(z[k]).decode("utf-8"))
            except Exception:                      # noqa: BLE001
                cfg = {}
            continue
        weights[k] = np.asarray(z[k], dtype=np.float32)
    return Checkpoint(weights, cfg, source=path)


# ---------------------------------------------------------------------------
# .TG（我们自己转换出来的格式）
# ---------------------------------------------------------------------------

def _tg_perm(nh, dh):
    """.TG 的交错 RoPE 布局：perm[2k]=k, perm[2k+1]=k+dh/2（与 gguf_to_tg.maybe_permute_rope 对齐）。"""
    perm = np.empty(dh, dtype=np.int64)
    for k in range(dh // 2):
        perm[2 * k] = k
        perm[2 * k + 1] = k + dh // 2
    return perm


def _tg_depermute(a, nh, dh):
    """把 .TG 的**交错**通道顺序还原成 HF 的 half 布局。

    为什么必须还原：`.TG` 里 q_proj/k_proj/q_norm/k_norm 的通道被转换器按交错 RoPE
    预置换过；用 half-rope 去算就会错到 rel≈0.44（实测），而且**不会报错**。
    实测反置换后与官方 HF 对齐到 1e-6。
    """
    inv = np.argsort(_tg_perm(nh, dh))
    if a.ndim == 1:
        return np.ascontiguousarray(a.reshape(nh, dh)[:, inv].reshape(-1))
    return np.ascontiguousarray(a.reshape(nh, dh, -1)[:, inv, :].reshape(a.shape))


def _expect_linear_shapes(cfg):
    """每个线性层**应该**是什么形状（.TG / HF 都是 [out, in] 约定）。"""
    D = int(cfg["D"])
    NH = int(cfg["NH"])
    NKV = int(cfg["KV"])
    DH = int(cfg["DH"])
    FF = int(cfg["FF"])
    return {
        "attn.q_proj.weight": (NH * DH, D),
        "attn.k_proj.weight": (NKV * DH, D),
        "attn.v_proj.weight": (NKV * DH, D),
        "attn.out_proj.weight": (D, NH * DH),
        "ffn.w1.weight": (FF, D),
        "ffn.w2.weight": (FF, D),
        "ffn.w3.weight": (D, FF),
        "lm_head.weight": (int(cfg["V"]), D),
    }


def fix_linear_layout(weights, cfg, strict=True):
    """校正/校验线性层布局，返回 (weights, 转置了哪些名字)。

    `.TG` 有两种可能的存储约定：
      * 标准 [out, in]  —— 与我们的 `linear(x, w) = x @ w.T` 一致；
      * 转置 [in, out]  —— 需要转置回来。
    形状正好是转置的，就自动转；**两者都不是**就报错（绝不静默放过）。
    """
    exp = _expect_linear_shapes(cfg)
    flipped = []
    out = dict(weights)
    # ★ 必须遍历**所有** key：一个 24 层模型有 24 个 blocks.N.ffn.w1，
    #   只修"第一个匹配到的"会漏掉其余 23 层（实测踩过：单测里立刻暴露）。
    for key in list(out):
        want = None
        for name, shape in exp.items():
            if key.endswith(name):
                want = shape
                break
        if want is None:
            continue
        a = out[key]
        got = tuple(int(x) for x in a.shape)
        if got == want:
            continue
        if got == want[::-1]:
            out[key] = np.ascontiguousarray(a.T)
            flipped.append(key)
            continue
        if strict:
            raise CheckpointError(
                "线性层形状对不上：%s 是 %s，期望 %s（或转置 %s）。"
                "修法：① 这是别的模型结构的权重吗？核对 config/头里的 "
                "d_model/n_heads/n_head_kv/n_ff；② 若是转置存储的 .TG，"
                "请确认转换器写的是 [out,in] 还是 [in,out]（本函数只接受这两种）"
                % (key, got, want, want[::-1]))
    return out, flipped


def norm_sanity(weights, lo=0.2, hi=3.0):
    """检查归一化权重是否"像"RMSNorm/LayerNorm 的 gamma（应≈1）。

    返回 (好坏统计, 可疑名单)。判据：1-D 且名字含 norm 的张量，
    均值应落在 [lo, hi]、且极值不应远离 1。

    为什么需要：实测（2026-11-01）某份 `.TG` 的 2-D 权重统计完全正常，
    但 `final_norm.weight` 的均值是 **7.14**、`blocks.0.ln1.weight` 是 **0.03** ——
    加载不报错、训练不报错，只是模型永远训不好。这类"静默读错"必须被点名。
    """
    # ★ 命名要同时认 "norm" 和 "ln1/ln2"：实测出问题的正是 blocks.0.ln1.weight，
    #   它不含 "norm" 字样 —— 只匹配 norm 会漏掉最该抓的那个张量。
    import re as _re
    pat = _re.compile(r"(^|\.)ln\d*(\.|$)")
    good, bad = 0, []
    for k, a in weights.items():
        if a.ndim != 1:
            continue
        if "norm" not in k and not pat.search(k):
            continue
        m = float(np.mean(a))
        lo_a, hi_a = float(np.min(a)), float(np.max(a))
        if lo <= m <= hi and lo_a > -2.0 and hi_a < 10.0:
            good += 1
        else:
            bad.append((k, m, lo_a, hi_a))
    return good, bad


def load_tg(path, max_layers=None, vocab_limit=None):
    """读 .TG → Checkpoint。max_layers / vocab_limit 用于"切片对拍"（省内存）。"""
    from ..read_tg import read_tg, load_tensor
    if not os.path.isfile(path):
        raise CheckpointError("找不到 .TG 文件：%s。修法：检查路径" % (path,))
    info = read_tg(path)
    h = info["header"]
    n_layers, D, NH = int(h["n_layers"]), int(h["d_model"]), int(h["n_heads"])
    NKV = int(h.get("num_kv_heads") or NH)
    q0 = load_tensor(info, "blocks.0.attn.q_proj.weight")
    if q0 is None:
        raise CheckpointError(
            ".TG 里没有 blocks.0.attn.q_proj.weight。修法：确认这是完整模型而不是部分权重")
    DH = int(q0.shape[0]) // NH          # .TG 头里没有 head_dim，只能从 q_proj 行数推
    has_qk_norm = load_tensor(info, "blocks.0.attn.q_norm.weight") is not None

    def get(name, nh=None):
        a = load_tensor(info, name)
        if a is None:
            return None
        if nh is not None:
            a = _tg_depermute(a, nh, DH)
        return a

    weights = {}
    emb = get("token_embedding.weight")
    if emb is None:
        raise CheckpointError(".TG 缺少 token_embedding.weight")
    weights["token_embedding.weight"] = (
        emb[:vocab_limit] if vocab_limit else emb)
    fn = get("final_norm.weight")
    if fn is not None:
        weights["final_norm.weight"] = fn
    lm = get("lm_head.weight")
    tie = lm is None
    if lm is not None:
        weights["lm_head.weight"] = lm[:vocab_limit] if vocab_limit else lm

    n = n_layers if max_layers is None else min(max_layers, n_layers)
    for i in range(n):
        b = "blocks.%d." % i
        for key, nh in (("ln1.weight", None), ("ln2.weight", None),
                        ("attn.q_proj.weight", NH), ("attn.k_proj.weight", NKV),
                        ("attn.v_proj.weight", None),
                        ("attn.out_proj.weight", None),
                        ("attn.q_norm.weight", 1), ("attn.k_norm.weight", 1),
                        ("ffn.w1.weight", None), ("ffn.w2.weight", None),
                        ("ffn.w3.weight", None)):
            a = get(b + key, nh)
            if a is None:
                continue
            if vocab_limit and key == "attn.out_proj.weight":
                pass
            weights[b + key] = a
        for bias in ("q_proj", "k_proj", "v_proj"):
            a = get(b + "attn.%s.bias" % bias)
            if a is not None:
                weights[b + "attn.%s.bias" % bias] = a

    cfg = {"arch": {1: "llama", 2: "mistral", 3: "qwen2"}.get(int(h["arch"]), "qwen2"),
           "n_layer": n, "n_embd": D, "n_head": NH, "n_head_kv": NKV,
           "head_dim": DH, "n_ff": int(h["d_ff"]),
           "vocab_size": int(vocab_limit or h["vocab"]),
           "rope_theta": float(h["rope_theta"] or 10000.0), "eps": 1e-6,
           "tie_embeddings": tie, "DH": DH, "QK_NORM": has_qk_norm,
           "QKV_BIAS": any(k.endswith("q_proj.bias") for k in weights),
           "TIE": tie}
    if has_qk_norm:
        cfg["arch"] = "qwen3"
    # ★ 布局自洽检查：形状不对就转置或报错，绝不让"能加载但训出错模型"发生
    cfg_for_shape = {"D": D, "NH": NH, "KV": NKV, "DH": DH,
                     "FF": int(h["d_ff"]), "V": int(cfg["vocab_size"])}
    weights, flipped = fix_linear_layout(weights, cfg_for_shape)
    if flipped:
        cfg["layout_transposed"] = flipped

    # 归一化权重体检：多数不像 gamma 时，说明这份 .TG 读错了（实测踩过）
    n_good, bad = norm_sanity(weights)
    if bad and len(bad) * 2 >= (n_good + len(bad)):
        raise CheckpointError(
            ".TG 的归一化权重不像 RMSNorm/LayerNorm 的 gamma（%d/%d 个可疑，例如 %s 均值 %.2f）。"
            "修法：① 用 tools/tg_audit.py 与 safetensors 对拍同一模型；"
            "② 确认转换器版本与 .TG 版本一致（本文件 version=%s）；"
            "③ 若是旧转换器写出的文件，请重新转换 —— 这种错误不会报错，只会让模型训不好"
            % (len(bad), n_good + len(bad), bad[0][0], bad[0][1], h.get("version")))
    if bad:
        cfg["suspicious_norms"] = [b[0] for b in bad]

    return Checkpoint(weights, cfg, source=path)
