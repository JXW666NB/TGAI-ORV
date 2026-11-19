"""
TGAI V3 模型 —— 统一可伸缩架构（SAVE V2 / world V2 共用）
==========================================================
升级自 `TGAI NLP/model.py`（V4 时代：8 层 × 256 维 / 4 专家 / RoPE θ=10K）。

本文件按《TGAI NB V3 架构设计》落地，实现 6 条核心：
  ① 混合注意力：每 4 层 = 3× 压缩稀疏 CSA + 1× 全局注意力（3:1 交替）
  ② MoE FFN：N 专家激活 K + **共享专家**（DeepSeek 式）+ **aux-loss-free 负载均衡**
  ③ Attention Residuals（AttnRes）：每 4 层做一次跨层门控残差聚合
  ④ RoPE θ=500K（长上下文原生支持）
  ⑤ LM Head 与 Embedding 权重共享
  ⑥ **专家扩容嫁接**（graft）：0.86B→1.5B / 3B，保留原有语言能力

★ 工程红线（V3 文档 §8，本文件严格遵守）
  · 全部用 PyTorch 原生算子拼接，**不引入任何自定义 kernel**
  · ROCm/DCU：flash_sdp 探测失败即关闭（沿用原文件逻辑）
  · 梯度检查点 use_reentrant=True（训练侧）

★ 尺寸谱（一套代码，改 Config 即换规模）
  W10M / W22M      —— TGAI world V2（ESP32-S3，词表 8K）
  SAVE_0_5B / SAVE_0_86B / SAVE_1_5B / SAVE_3B —— TGAI SAVE V2 系列（词表 96K）
  V3_7B            —— 论文档目标（8 专家激活 2，词表 96K）

作者：TGAI / JXW    日期：2026-10-14
"""

import math
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
# ★★ 2026-10-16 修（P0-C，实测确认）★★
#   `torch.utils.checkpoint` 是**子模块**，不是 `torch.utils` 的属性：
#   实测 `import torch; hasattr(torch.utils, 'checkpoint')` → False（torch 2.6.0+cu124）。
#   本文件在 forward 里用 `torch.utils.checkpoint.checkpoint(...)`（3 处），
#   所以必须显式 import 该子模块，否则 `--grad-ckpt 1`（默认值！）直接
#   AttributeError: module 'torch.utils' has no attribute 'checkpoint'。
#   ★ 只在显式 import 之后才生效，因此这里是**唯一**可靠的修法。
import torch.utils.checkpoint

# ═══════════════════════════════════════════════════════════
# 特殊 token（与 tokenizer / 词表保持一致）
# ═══════════════════════════════════════════════════════════
# ★★ 2026-10-15 修正（实测抓到的严重 bug）★★
#   词表的真实布局是：
#       id 0..255    字节 token <b0>..<b255>（id == 字节值，映射简单）
#       id 256       <pad>
#       id 257       <unk>
#       id 258       <bos>
#       id 259       <eos>
#       id 260       <mask>
#       id 261       <|endoftext|>
#       id 262..264  <|fim_*|>
#       id 265..273  TGAI 9 个对话标记
#   而这里原先是 PAD_ID=0/UNK_ID=1/BOS_ID=2/EOS_ID=3 ——
#   **全部指向字节 token**（0x00/0x01/0x02/0x03）！后果：
#     · padding_idx=0 ⇒ 字节 0x00 的 embedding 被强制清零，
#       而真正的 <pad> 完全没被特殊处理
#     · generate(eos_token_id=3) ⇒ **遇到字节 0x03 就停止生成**
#     · encode(add_special=True) ⇒ 首尾加的是字节 token 而非 BOS/EOS
#   现按词表实际布局修正为 256/257/258/259。
PAD_ID = 256
UNK_ID = 257
BOS_ID = 258
EOS_ID = 259

# ── 兼容补丁（沿用 TGAI NLP/model.py）：ROCm 探测 flash SDPA ──
if torch.cuda.is_available() and getattr(torch.version, 'hip', None):
    try:
        _q = torch.randn(1, 4, 128, 64, device='cuda', dtype=torch.bfloat16)
        F.scaled_dot_product_attention(_q, _q, _q, is_causal=True)
        del _q
        print("[SDPA] ROCm flash attention 可用, 已启用")
    except Exception:
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            print("[SDPA] ROCm flash 不可用, 已关闭 (回退 math/mem_efficient)")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════
@dataclass
class TGAIV3Config:
    """V3 配置。字段分四组：基础 / 注意力 / MoE / 长上下文。"""

    # ── 基础 ──
    vocab_size: int = 96000
    d_model: int = 1024
    n_layers: int = 20
    n_heads: int = 16
    d_ff: int = 2816                 # 每个专家的中间维度（SwiGLU）
    max_seq_len: int = 65536         # 64K 原生
    dropout: float = 0.0
    pad_token_id: int = PAD_ID
    tie_embeddings: bool = True
    norm_eps: float = 1e-6

    # ── 注意力（混合：CSA 稀疏层 + 全局层 3:1）──
    attn_global_every: int = 4       # 每 4 层里有 1 层是全局注意力
    n_kv_heads: int = 0              # 全局层 GQA 的 KV 头数（0 = 等于 n_heads，即 MHA）
    csa_compress: int = 4            # CSA 的序列压缩率
    csa_latent: int = 0              # CSA 的 KV 潜在维度（0 = d_model // 4）
    csa_conv_kernel: int = 3         # CSA 压缩后的深度可分离卷积核
    csa_local_window: int = 0        # CSA 局部精确窗口（0 = 自动取 8×压缩率）
    attn_qk_norm: bool = True        # QK-Norm（稳定大 lr；★ 注意 K2 报告：与 MLA 不兼容，我们非 MLA）
    # ★★ 2026-10-16 新增（P0-A 修复配套）：CSA 跨文档隔离 ★★
    #   True（默认）= 给 CSA 传 seg_mask 并做「边界感知分段池化 + 块对角掩码」。
    #     · 全局压缩分支：query 只能看见**同文档**的 chunk（含因果）
    #     · 局部精确窗分支：query 只能看见**同文档**的窗口内 key（修反向泄漏）
    #     · 分段池化：chunk 边界按文档边界对齐，避免"chunk 内混着前一篇文档的尾巴"
    #   False = 退回修复前行为（纯因果、固定 chunk 池化），仅供 A/B 对照。
    #   ★ seg_mask=None（推理/引擎路径）时该开关无效果，行为与修复前逐位一致。
    csa_seg_isolate: bool = True
    # ★★ 2026-10-16 新增（P0-A 修复配套②）：CSA 压缩卷积的因果性 ★★
    #   True（默认）= 只左填充 k-1，位置 j 只依赖 j-k+1..j（真因果，跨文档不泄漏）。
    #   False        = 退回 `nn.Conv1d(padding=k-1)` 的双向填充（含未来 chunk）。
    #   ★ 旧代码是双向填充 + 切左侧，实测会让末 2 个 chunk 看见未来 2 个 chunk
    #     （改 chunk 0..7 的输入 ⇒ chunk 8 输出变 1.450e+00）。这是真 BUG，
    #     默认已修正；保留开关便于 A/B 与回退。
    csa_causal_conv: bool = True
    # ★★ 2026-10-16 新增（P1-C）：QK-Clip 遥测的采样间隔 ★★
    #   实测（加速方案 §1.5）：`return_qk_max=True` 让 fwd+bwd 从 1.144 s → 1.241 s
    #   （+8.5%），代价是每层多算一次 fp32 的 q@kᵀ（B=8,T=2048 时约 2.1 GB 临时显存）。
    #   1 = 每步都算（= 修复前行为，默认不变）；N>1 = 每 N 步算一次，其余步复用上次结果。
    #   ★ 监控能力保留：QKmax 列仍每步有值（非采样步沿用上一次的统计）。
    #   由 Trainer 在每个 forward 前设置 `model._qk_sample = (step % N == 0)`。
    qk_telemetry_every: int = 1
    # ★★ 2026-10-16 新增（batch 扫描用）：分块交叉熵 ★★
    #   0 = 关闭（= 修复前行为，一次性算 (B,T-1,96K) 的 CE）
    #   >0 = 按该 chunk 长度分段算 CE 再累加，峰值只需 1 个 chunk 的 logits。
    #   ★ 数学等价：sum(各 chunk 的 sum-CE) / 元素总数 == 一次性 mean-CE
    #     （F.cross_entropy(reduction='sum') 是按**元素总数**求和的，与 chunk 无关）
    ce_chunk: int = 0
    qk_clip_tau: float = 100.0       # ★ QK-Clip 阈值 τ（Kimi K2 用 100；0 = 关闭）
    qk_clip_alpha: float = 0.5       # QK-Clip 的 Q/K 分配指数（K2 用 0.5）

    # ── MoE ──
    n_experts: int = 8               # 专家池（V3 文档：8 专家激活 2）
    n_activated: int = 2             # 每 token 激活专家数
    n_shared_experts: int = 1        # 共享专家（DeepSeekMoE，**直接相加无门控**）
    moe_bias_lr: float = 0.001       # aux-loss-free 偏置更新速度 γ（符号式更新）
    moe_router_z_loss: float = 0.0001  # 路由 logits 稳定正则
    moe_aux_alpha: float = 0.0001    # 互补序列辅助损失权重（V3 里极小）
    moe_balance: str = 'quantile'    # 'quantile'(K3 QB，无 γ 超参) | 'sign'(DeepSeek aux-free)
    # ★ 2026-10-16 新增（P1-D）：MoE 专家分组实现，两种**数学等价**
    #   'mask'  （默认，= 修复前行为）：布尔索引逐专家取子集
    #   'sorted'：argsort 把 token 按专家排好 → 连续切片（实测单层 MoE fwd+bwd
    #             N=8192 时 0.1425 s → 0.1113 s，1.28×，加速方案 §2.3）
    #   ★ 改默认前必须在 10M 预设上 A/B：两者 loss 曲线必须一致。
    moe_impl: str = 'mask'
    dense_ffn: bool = False          # True = 不用 MoE，纯稠密 FFN（极小模型用）
    # ★ K3 落地三点（研究复命 2026-10-14）：
    #   ① 专家用 SiTU-GLU（有界，fp16 刚需）
    #   ② routed 聚合后加 RMSNorm（原文称稳定改善 val loss，不只防爆）
    #   ③ 路由用 Quantile Balancing（去掉 γ 超参）
    moe_use_situ: bool = True        # 专家 FFN 用 SiTU-GLU
    moe_post_norm: bool = True       # 聚合后 RMSNorm

    # ── AttnRes（K3 Block AttnRes，arXiv:2603.15031）──
    attn_res_block: int = 4          # ★ 每多少个 Transformer 层构成一个 block（论文建议总量 ~8 block）
    attn_res_enabled: bool = True
    # ★ 小模型（<50M）关掉 AttnRes 更划算：层少时深度注意力收益低，却增加实现复杂度
    dense_attn_res_free: bool = True

    # ── 层共享（MobileLLM immediate block-wise weight sharing）──
    #   ★ ESP32-S3 研究结论（2026-10-14，联网核实）：
    #     "把共享权重放 cache 里立刻算两遍，可避免权重在 SRAM 与 DRAM 间搬运"
    #     —— 正是 PSRAM 带宽受限（实测 58-61 MB/s）设备最需要的性质：
    #     同样 5.35MB 权重换 2 倍等效深度，存储/带宽成本为 0（代价：计算翻倍）
    #   实现：每 share_repeat 层复用同一组 block 权重（层索引 % share_group）
    layer_share_group: int = 0       # 0 = 关闭；N>0 = 每 N 个"物理层"组成一组共享权重

    # ── RoPE / 长上下文 ──
    rope_theta: float = 500000.0     # ★ V3 文档：θ=500K
    rope_scaling: float = 1.0
    self_extend: bool = False
    hard_max_seq: int = 0            # 生成时的硬上限（0 = 用 max_seq_len）
    # ★ 异构 RoPE（星火 X2.5-4B 的方案，2026-10-14 联网核实 config.json 逐字）
    #   layer_types = [滑,滑,滑,全局]×9；两种层用**完全不同**的位置编码：
    #     全局层: partial_rotary_factor=0.25, rope_theta=5e6  → 只旋转 1/4 维度
    #     滑窗层: partial_rotary_factor=1.0,  rope_theta=1e4  → 全维旋转
    #   ⇒ 两个位置信号各管一个尺度，互不冲突（绕开 KDA/NoPE 与 RoPE 冲突的坑）
    hetero_rope: bool = False        # 开启后按下面两个字典覆盖每层 RoPE 设置
    rope_theta_global: float = 5000000.0
    rope_theta_local: float = 10000.0
    partial_rotary_global: float = 0.25   # 全局层只旋转这么比例的维度
    partial_rotary_local: float = 1.0
    # ★ 滑窗注意力（星火 X2.5：27 层窗口 512 + 9 层全局，KV cache 砍到 ~1/4）
    sliding_window: int = 0          # 0 = 关闭；>0 = CSA 层改用滑窗掩码

    @property
    def d_k(self) -> int:
        """每个注意力头的维度。n_heads 必须整除 d_model；d_k 自动取偶数。"""
        d = self.d_model // self.n_heads
        return d if d % 2 == 0 else d - 1

    @property
    def n_q_heads(self) -> int:
        return self.d_model // self.d_k

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads > 0 else self.n_q_heads

    @property
    def latent(self) -> int:
        return self.csa_latent if self.csa_latent > 0 else max(64, self.d_model // 4)

    def is_global_layer(self, i: int) -> bool:
        """第 i 层（0-based）是否为全局注意力层。

        ★ 注意：这是**方法**不是属性（需要层索引参数）。
          历史坑：误加 @property 后变成属性，调用 `cfg.is_global_layer(i)` 会报
          "missing 1 required positional argument"，且错误信息会误导。
        """
        return (i % self.attn_global_every) == (self.attn_global_every - 1)

    def rope_cfg(self, is_global: bool) -> Tuple[float, float]:
        """返回该层类型的 (theta, partial_rotary_factor)。

        ★ 异构 RoPE（星火 X2.5 方案，2026-10-14 核实官方 config.json）：
            全局层: partial=0.25, θ=5e6  → 长程近 NoPE，只保短程相对位置
            滑窗层: partial=1.0,  θ=1e4  → 短程满分辨率，无需外推
          未开启 hetero_rope 时，两层都用 config.rope_theta 与全旋转。
        """
        if not self.hetero_rope:
            return self.rope_theta, 1.0
        if is_global:
            return self.rope_theta_global, self.partial_rotary_global
        return self.rope_theta_local, self.partial_rotary_local

    @property
    def use_csa(self) -> bool:
        """是否启用压缩稀疏注意力。

        ★ ESP32-S3 研究结论（2026-10-14）：小模型应**砍掉 CSA** ——
          attention 只占 24% 算力，压缩一半也只省 12%；但稀疏/gather 访存
          会踩 PSRAM 随机读慢 5-7× 的痛点。故 csa_compress<=1 时退化为标准因果注意力。
        """
        return self.csa_compress > 1


# ═══════════════════════════════════════════════════════════
# RoPE（θ 可配；沿用原实现并去掉不必要的 self-extend 分支复杂度）
# ═══════════════════════════════════════════════════════════
class RotaryPositionEmbedding(nn.Module):
    """RoPE，支持**部分旋转**（partial rotary，星火 X2.5 方案）+ 可配 θ。

    部分旋转（核实自星火 X2.5 官方 modeling_spark.py）：
        x_rot  = x[..., :rope_head_dim]      # 只旋转前 rope_head_dim 维
        x_pass = x[..., rope_head_dim:]      # 其余原样通过（等价 NoPE）
        out = cat(rope(x_rot), x_pass)
    用途：全局层用 partial=0.25 + θ=5e6（长程近 NoPE，只保短程相对位置）；
          滑窗层用 partial=1.0 + θ=1e4（短程满分辨率，无需外推）。
    """

    def __init__(self, dim: int, max_seq_len: int, theta: float = 500000.0,
                 scaling: float = 1.0, partial: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling = scaling
        # 旋转维数取偶数
        rd = int(round(dim * partial))
        rd = rd if rd % 2 == 0 else rd - 1
        self.rot_dim = max(2, min(rd, dim))
        self.partial = self.rot_dim < dim

        freqs = 1.0 / (theta ** (torch.arange(0, self.rot_dim, 2).float() / self.rot_dim))
        positions = torch.arange(max_seq_len).float() / scaling
        angles = torch.outer(positions, freqs)              # (L, rot_dim/2)
        emb = angles.repeat_interleave(2, dim=-1)           # (L, rot_dim)
        self.register_buffer('cos', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin', emb.sin().unsqueeze(0).unsqueeze(0))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """交错取反：(x1,x2,x3,x4) → (-x2,x1,-x4,x3)"""
        r = torch.empty_like(x)
        r[..., ::2] = -x[..., 1::2]
        r[..., 1::2] = x[..., ::2]
        return r

    def _cos_sin(self, T: int, offset: int, device, dtype):
        if offset + T <= self.max_seq_len:
            return (self.cos[:, :, offset:offset + T], self.sin[:, :, offset:offset + T])
        pos = torch.arange(offset, offset + T, device=device).float() / self.scaling
        freqs = 1.0 / (self.theta ** (torch.arange(
            0, self.rot_dim, 2, device=device).float() / self.rot_dim))
        emb = torch.outer(pos, freqs).repeat_interleave(2, dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None,
                offset: int = 0) -> torch.Tensor:
        """x: (B, H, T, D)"""
        T = x.shape[-2]
        if position_ids is not None:
            pos = position_ids.reshape(-1).float() / self.scaling
            freqs = 1.0 / (self.theta ** (torch.arange(
                0, self.rot_dim, 2, device=x.device).float() / self.rot_dim))
            emb = torch.outer(pos, freqs).repeat_interleave(2, dim=-1)
            cos = emb.cos().view(1, 1, -1, self.rot_dim)
            sin = emb.sin().view(1, 1, -1, self.rot_dim)
        else:
            cos, sin = self._cos_sin(T, offset, x.device, x.dtype)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        if not self.partial:
            return x * cos + self._rotate_half(x) * sin
        # ★ 部分旋转：只动前 rot_dim 维
        xr = x[..., :self.rot_dim]
        xp = x[..., self.rot_dim:]
        xr = xr * cos + self._rotate_half(xr) * sin
        return torch.cat([xr, xp], dim=-1)


# ═══════════════════════════════════════════════════════════
# RMSNorm / SwiGLU
# ═══════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ★ 提速优化 2026-10-19（V100 提速②）：单遍求范数，去掉 (N,D) 级 fp32 物化。
        #   旧实现 `x.float().pow(2).mean(...)` 物化两份 (N,D) fp32 临时张量
        #   （x.float() 一次 + pow(2) 一次），fp16 输入时每元素访存约 46 B；
        #   反向还要为 pow 存一份 fp32 副本。现用 vector_norm 单遍规约
        #   （fp16 输入内核以 fp32 累加、输出 fp32），访存约 10 B/elem。
        #   公式不变：rms = rsqrt(mean(x²) + eps)；mean 用 n²/D 等价还原。
        #   数值偏差 ~1e-6 量级（规约顺序差），远低于 fp16 量化噪声 5e-4，
        #   loss 曲线 A/B 可过；clamp_min 仅防 x=0 行的 0/0 梯度（前向不受影响）。
        n = torch.linalg.vector_norm(x, 2, -1, keepdim=True,
                                     dtype=torch.float32).clamp_min(1e-12)
        rms = torch.rsqrt(n * n / x.shape[-1] + self.eps)
        return (x * rms).type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class SiTUGLUFFN(nn.Module):
    """★ SiTU-GLU（Kimi K3 Eq.12）—— 有界激活，为低精度训练而设。

    论文动机（原文）：
      "both multiplicative factors in SwiGLU are **unbounded** … this increases
       the risk of overflow in low-precision arithmetic"
    公式（β₁ = 4 gate 分支 / β₂ = 25 up 分支，输出上界 |f| ≤ β₁·β₂ = 100）：
        SiTU-GLU(x) = [β₁·tanh(W_g x / β₁) ⊙ Sigmoid(W_g x)] ⊙ [β₂·tanh(W_u x / β₂)]
    与 hard-clip 的区别：tanh 在原点近似线性 ⇒ 局部行为贴近 SwiGLU，不损失表达能力。

    ★ 为什么我们必须用（V100 实测事实）：
      V100 无 BF16 张量核心 ⇒ 训练只能 FP16（上界 65504）或 FP32。
      SwiGLU 的 SiLU(x)·(Wx) 两项都可无界增长，大 lr / 深层堆叠时容易溢出成 inf ⇒ NaN。
      SiTU-GLU 把乘积硬压到 ≤100，从源头消除这个隐患。
    """

    def __init__(self, d_model: int, d_ff: int, beta1: float = 4.0, beta2: float = 25.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # gate 分支
        self.w2 = nn.Linear(d_model, d_ff, bias=False)   # up 分支
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.b1 = beta1
        self.b2 = beta2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.w1(x)
        u = self.w2(x)
        gate = (self.b1 * torch.tanh(g / self.b1)) * torch.sigmoid(g)
        up = self.b2 * torch.tanh(u / self.b2)
        return self.w3(gate * up)


# ★★ 2026-10-18 新增（根因修复）：专家输入的 8 元素对齐 pad ★★
#   背景：mask/sorted 两个 MoE 分支都把 [N*top_k, d] 按专家切成动态长度子张量
#         再送进 experts[eid]。实测在 V100 + Windows + TCC 上，当子张量行数
#         是奇数时，cuBLAS 的 fp16 Tensor Core kernel 会走到未对齐的向量化
#         load 路径，抛 CUBLAS_STATUS_EXECUTION_FAILED / misaligned address。
#   修法：把行数 pad 到 8 的倍数（fp16 的 128-bit 向量宽度 = 8 个元素），
#         算完再切回原长度。pad 行用最后一行复制填充，切片后即丢弃。
#   ★ 数学等价：只改变送进 GEMM 的行数，输出切片后与原来完全一致。
def _pad_to_multiple(x, mult=8):
    n = x.shape[0]
    r = n % mult
    if r == 0 or n == 0:
        return x, n
    pad = mult - r
    tail = x[-1:].expand(pad, *x.shape[1:])
    return torch.cat([x, tail], dim=0), n


def _sdpa_with_max(q, k, v, attn_mask, need_max):
    """SDPA 包装：可选返回每头 pre-softmax 最大分数（QK-Clip 遥测用）。

    ★ 为什么单独算 max：Kimi K2 报告指出，Muon 训练下个别 head 的 pre-softmax
      注意力分数会涨到 >1000，导致 loss spike；QK-Clip 需要**逐 head 的最大值**
      （均值/RMS 正常时单个 head 也可能跑飞）。
    ★ 只在 need_max=True 时才算（多一次 q@kᵀ，训练开销可接受；推理时关闭）。
    ★★ 2026-10-16 修（父 agent 实测抓到的 bug）★★
      原来还要求 `q.requires_grad`。但 `use_reentrant=True/False` 的 checkpoint
      **首次前向是在 no_grad 下跑**的（探针里能看到
      "None of the inputs have requires_grad=True" 警告）⇒ q.requires_grad=False
      ⇒ 直接早退返回 head_max=None ⇒ **grad_ckpt 下 QK 遥测恒为 0**。
      实测：B=1..4 + grad_ckpt=1 时 QK 非零层数=0；同配置 grad_ckpt=0 时 16 层/max=42.5。
      而 grad_ckpt 是**默认开启**的 ⇒ QK 监控（K2 报告的核心稳定性指标）完全失效。
      修法：need_max=True 就算，不看 requires_grad（块内本来就是 no_grad，无副作用）。
    """
    if not need_max:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                              dropout_p=0.0, is_causal=False), None
    with torch.no_grad():
        try:
            scores = torch.matmul(q.detach().float(), k.detach().float().transpose(-2, -1))
            if attn_mask is not None:
                # ★ 2026-10-16 修（父 agent 实测定位）：掩码有 2D (T,K) 与 4D (B,1,T,K) 两种形态。
                #   原写法对 4D 掩码会变成 (1,1,B,1,T,K)，与 scores (B,H,T,K) 广播成 6 维
                #   ⇒ head_max 语义错（batch>1 时取到的是 batch0 的值）
                #   ⇒ 训练日志里 QKmax 这个核心稳定性指标不可信。
                mk = attn_mask.float()
                if mk.dim() == 2:
                    mk = mk.unsqueeze(0).unsqueeze(0)     # (T,K)   -> (1,1,T,K)
                elif mk.dim() == 3:
                    mk = mk.unsqueeze(1)                  # (B,T,K) -> (B,1,T,K)
                # 4D 已是 (B,1,T,K)，可直接与 scores 广播
                scores = scores + mk
            scores = scores.masked_fill(torch.isinf(scores) & (scores < 0), float('-inf'))
            head_max = scores.amax(dim=(-1, -2))          # (B, H)
            head_max = head_max.amax(dim=0)               # (H,)
        except (IndexError, ValueError) as _e:
            # ★ 2026-10-18 修（架构审查定位）：原为裸 `except Exception`，
            #   会把 CUDA error / OOM 一起吞掉。CUDA 错误是粘滞的，被吞后
            #   错误状态不清除，下一个算子才爆 ⇒ 报错位置漂移、真凶被掩盖。
            #   现只兜底形状类错误（IndexError/ValueError），CUDA 错误一律上抛。
            head_max = None
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                         dropout_p=0.0, is_causal=False)
    return out, head_max


# ═══════════════════════════════════════════════════════════
# 注意力层 1：全局 GQA（分组查询注意力）
# ═══════════════════════════════════════════════════════════
class GlobalAttention(nn.Module):
    """标准因果注意力，支持 GQA（n_kv_heads < n_q_heads）+ KV Cache + QK-Clip 遥测。

    ★ V100 实测（2026-10-14）：SDPA 的显式 `attn_mask`（含非方 mask）与
      `enable_gqa` 在 SM 7.0 上全部可用，因此这里用显式掩码路径（更可控）。
    """

    def __init__(self, config: TGAIV3Config, is_global: bool = True):
        super().__init__()
        self.cfg = config
        self.is_global_flag = is_global
        self.n_q = config.n_q_heads
        self.n_kv = config.kv_heads
        self.d_k = config.d_k
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.q_proj = nn.Linear(config.d_model, self.n_q * self.d_k, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv * self.d_k, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv * self.d_k, bias=False)
        self.o_proj = nn.Linear(self.n_q * self.d_k, config.d_model, bias=False)
        # ★ 异构 RoPE：按层类型取 (theta, partial)
        _th, _pr = config.rope_cfg(is_global)
        self.rope_theta, self.rope_partial = _th, _pr
        self.rope = RotaryPositionEmbedding(self.d_k, config.max_seq_len,
                                            _th, config.rope_scaling, _pr)
        if config.attn_qk_norm:
            self.q_norm = RMSNorm(self.d_k, config.norm_eps)
            self.k_norm = RMSNorm(self.d_k, config.norm_eps)

    def forward(self, x: torch.Tensor, cache: Optional[Dict] = None,
                use_cache: bool = False, return_attn_max: bool = False,
                seg_mask: Optional[torch.Tensor] = None):
        """seg_mask: (B, T, T) bool —— True = 可见（块对角，跨文档互不可见）。

        ★ 打包训练的关键（研究标★最高风险项）：只重置 position_ids 不够，
          必须真的让不同文档互相看不见，否则长上下文阶段明显掉分且不报错。
        """
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_q, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.d_k).transpose(1, 2)

        if self.cfg.attn_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        past = 0
        if cache is not None:
            past = cache.get('len', 0)
            if cache.get('k') is not None:
                k = torch.cat([cache['k'], k], dim=2)
                v = torch.cat([cache['v'], v], dim=2)
            if use_cache:
                cache['k'], cache['v'] = k, v
                cache['len'] = past + T

        q = self.rope(q, offset=past)
        k = self.rope(k, offset=past)

        if self.n_kv != self.n_q:
            rep = self.n_q // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        klen = k.shape[2]
        attn_mask = None
        sw = self.cfg.sliding_window
        use_sw = bool(sw and sw > 0 and not self.is_global_flag and past == 0)
        if seg_mask is not None and past == 0:
            # ★ 块对角：直接从 seg_mask 生成（bool → -inf/0），叠加滑窗
            #
            # ★★ 2026-10-16 修（P0-B，实测确认的严重正确性缺陷）★★
            #   原实现：
            #       m = seg_mask
            #       if m.dim() == 4: m = m[:, 0]      # (B,1,T,T) → (B,T,T)
            #       ...
            #       attn_mask.masked_fill_(~m[0] if m.dim() == 3 else ~m, -inf)
            #                                              #   ↑↑ 只取 batch 0！
            #   make_seg_mask 返回 (B,1,T,T) ⇒ 取 [:,0] 得 (B,T,T)，随后又只取 m[0]
            #   得到 (T,T)，被 SDPA 广播到整个 batch ⇒ **batch 1..B-1 全用 batch 0 的
            #   文档边界**：该隔开的没隔开（跨文档泄漏），不该隔开的被隔开（文档内截断）。
            #   实测（B=2，batch0 分界=8 / batch1 分界=24）：
            #       vs 正确 4D 掩码      最大差 = 9.748e-01
            #       vs 只用 m[0] 的复刻  最大差 = 1.490e-07  ← 就是它
            #       batch1 第 28 行可见 key = [8..28]（21 个），正确应为 [24..28]（5 个）
            #   修法：保持 batch 维，用 (B,1,T,T) 的 4D 掩码（mem_efficient 后端实测支持）。
            m = seg_mask
            if m.dim() == 4:
                m = m[:, 0]                       # (B,1,T,T) → (B,T,T)
            if m.dim() == 2:
                m = m.unsqueeze(0)                # (T,T) → (1,T,T)
            if m.shape[0] == 1 and B > 1:
                m = m.expand(B, -1, -1)           # 广播到 batch（单样本掩码的兼容路径）
            if use_sw:
                qi = torch.arange(T, device=x.device).unsqueeze(1)
                kj = torch.arange(T, device=x.device).unsqueeze(0)
                m = m & (kj >= (qi - sw + 1))
            # ★ 一次性算出 (B,1,T,T)，让 SDPA 按 batch 维取用（不再广播 batch0）
            attn_mask = torch.zeros(B, 1, T, m.shape[-1], device=x.device,
                                    dtype=q.dtype)
            attn_mask.masked_fill_(~m.unsqueeze(1), float('-inf'))
        elif T > 1:
            attn_mask = torch.full((T, klen), float('-inf'),
                                   device=x.device, dtype=q.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=klen - T + 1)
            if use_sw:
                qi = torch.arange(T, device=x.device).unsqueeze(1) + past
                kj = torch.arange(klen, device=x.device).unsqueeze(0)
                attn_mask.masked_fill_(kj < (qi - sw + 1), float('-inf'))
        out, head_max = _sdpa_with_max(q, k, v, attn_mask, return_attn_max)
        out = out.transpose(1, 2).reshape(B, T, self.n_q * self.d_k)
        return self.o_proj(out), head_max


# ═══════════════════════════════════════════════════════════
# 注意力层 2：压缩稀疏 CSA（Compressed Sparse Attention）
# ═══════════════════════════════════════════════════════════
class CompressedSparseAttention(nn.Module):
    """CSA —— 用「低秩压缩 + 序列压缩 + 局部深度卷积」近似长程注意力。

    设计（V3 文档 §3「CSA/HCA 轻量版」+ §11「不用自定义 kernel」）：
        K,V → 低秩下投影 (d → latent)
            → 序列压缩（把连续 c 个位置合并，均值池化）  ★ 降 FLOPs 与 KV 长度
            → 因果深度可分离卷积（补回局部邻域信息）
            → RMSNorm
            → 全注意力（对压缩后序列）
            → 低秩上投影回 (latent → d)
    输出再与「局部精确窗口注意力」相加，保证近邻 token 的精确交互不丢。

    ★ 复杂度：O(T·T/c)，c=4 时约降 75% 的长程注意力开销。
    """

    def __init__(self, config: TGAIV3Config):
        super().__init__()
        self.cfg = config
        d = config.d_model
        self.latent = config.latent
        self.c = max(2, config.csa_compress)
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.n_q = config.n_q_heads
        self.n_kv = config.kv_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Q 走正常投影；K/V 走低秩压缩
        self.q_proj = nn.Linear(d, self.n_q * self.d_k, bias=False)
        self.kv_down = nn.Linear(d, self.latent, bias=False)
        self.k_down = nn.Linear(self.latent, self.n_kv * self.d_k, bias=False)
        self.v_down = nn.Linear(self.latent, self.n_kv * self.d_k, bias=False)
        self.o_proj = nn.Linear(self.n_q * self.d_k, d, bias=False)

        # 压缩后的局部卷积（因果）+ 归一
        # ★★ 2026-10-16 修（P0-A 第二个泄漏源）★★
        #   原来这里是 `padding=config.csa_conv_kernel - 1`（**双向**填充），
        #   配合 `[:, :, :n_out]` 切片 ⇒ 位置 j 的窗是 j-(k-1)..j+(k-1)，含未来。
        #   现在 padding 交给 `_conv()` 显式控制（默认只左填充，真因果）。
        #   ★ 踩过的坑：只改 `_conv` 加左填充、忘了这里还有 padding=k-1
        #     ⇒ 总填充 2(k-1)，窗变成 j-2..j+2，问题原样存在（实测差 1.052e+00）。
        self.conv = nn.Conv1d(self.latent, self.latent, kernel_size=config.csa_conv_kernel,
                              padding=0, groups=self.latent, bias=False)
        self.latent_norm = RMSNorm(self.latent, config.norm_eps)

        # 局部精确窗口（保近邻交互）
        win = config.csa_local_window if config.csa_local_window > 0 else max(32, self.c * 8)
        self.local_win = win
        # ★ 异构 RoPE：CSA 层按「非全局层」取配置（滑窗层 = 全旋转 + 标准 θ）
        _th, _pr = config.rope_cfg(False)
        self.rope_theta, self.rope_partial = _th, _pr
        self.l_rope = RotaryPositionEmbedding(self.d_k, config.max_seq_len,
                                              _th, config.rope_scaling, _pr)
        self.g_rope = RotaryPositionEmbedding(self.d_k, config.max_seq_len,
                                              _th, config.rope_scaling, _pr)
        self.gate = nn.Parameter(torch.zeros(1))     # 局部/全局混合门（0 起步=先信任全局）

        if config.attn_qk_norm:
            self.q_norm = RMSNorm(self.d_k, config.norm_eps)
            self.k_norm = RMSNorm(self.d_k, config.norm_eps)

    def _compress(self, h: torch.Tensor, seg: Optional[torch.Tensor] = None,
                  cid: Optional[torch.Tensor] = None,
                  n_chunk: Optional[torch.Tensor] = None,
                  pos: Optional[torch.Tensor] = None,
                  single: bool = False) -> torch.Tensor:
        """h: (B, T, latent) → (B, Tc, latent)，Tc = ceil(T/c) **恒定不变**。

        seg=None：按固定 chunk（每 c 个 token 一组）均值池化，尾部复制补齐。
        seg/cid/n_chunk 给出：**按文档边界分段池化**（每篇文档独立切 chunk，首尾相接），
          末尾补零到 Tc 个 chunk（无效 chunk 由掩码屏蔽）。

        ★★ 为什么必须两件事一起做（P0-A 实测依据）★★
          ① 只加掩码不够：固定 chunk 会把两篇文档的 token 池进同一个 chunk，
             文档B 的 token 就**看不到自己文档开头那几个 token**（被一起屏蔽了）
             —— 权重看起来是对的，但被池化的内容本身脏了。
          ② conv 的填充方式本身在泄漏（见 `_conv`）：`padding=k-1` 是双向填充，
             切前 Tc 个位置后每个位置还看得见**未来 2 个 chunk**。实测：把 chunk 0..7
             的输入全换掉，chunk 8 的卷积输出变化 1.450e+00、chunk 9 变化 8.060e-01
             —— 而 chunk 8/9 的输入一个字都没动。
        """
        B, T, L = h.shape
        c = self.c
        if seg is None:
            pad = (c - T % c) % c
            if pad:
                h = torch.cat([h, h[:, -1:, :].expand(B, pad, L)], dim=1)
            Tp = T + pad
            h = h.view(B, Tp // c, c, L).mean(dim=2)     # 均值池化压缩
            h = h.transpose(1, 2)                        # (B, L, T/c)
            h = self._conv(h, Tp // c)
            h = h.transpose(1, 2)                        # (B, T/c, L)
            return self.latent_norm(h)

        # ── 边界感知分段池化（逐样本；训练时 B 小，开销可忽略）──
        if cid is None or n_chunk is None or pos is None:
            raise ValueError('seg 给定时必须同时给 cid / n_chunk / pos')
        Tc = (T + c - 1) // c
        parts = []
        for b in range(B):
            # ★ 提速优化 2026-10-19（V100 提速③a）：零同步分段池化。
            #   旧代码每样本 3 次 D2H（nb.item() + 两次布尔 gather），4 样本 × 12 层
            #   × 48 micro × 2(重算) ≈ 每步 7 千次。新写法「无效位置 ×0」：
            #   cid 无效位=0、贡献 h×0、计数 +0 ⇒ 与逐样本过滤**逐位相同**，全程无同步。
            #   空 chunk 计数 0 → clamp_min(1) → 补零行/1 = 0，同旧分支语义。
            valid_f = (seg[b] >= 0).to(h.dtype).unsqueeze(-1)     # (T,1)
            cbi = cid[b].clamp(0, Tc - 1).to(torch.long)          # (T,) 无效位=0
            acc = torch.zeros(Tc, L, device=h.device, dtype=h.dtype)
            acc = acc.index_add(0, cbi, h[b] * valid_f)
            cnt = torch.zeros(Tc, device=h.device, dtype=torch.float32)
            cnt = cnt.index_add(0, cbi, valid_f.squeeze(-1).float())
            acc = acc / cnt.clamp_min(1).unsqueeze(-1).to(acc.dtype)
            parts.append(acc)
            self._dbg_chunks = getattr(self, '_dbg_chunks', 0) + 1   # ★插桩（口径：样本）
        hc = torch.stack(parts, dim=0)                    # (B, Tc, L)
        # ★★ 分段感知卷积（P0-A 的第三个、也是最隐蔽的泄漏源）★★
        #   卷积是**因果**的（位置 j 只读 j-k+1..j），因此文档B 的第一个 chunk 会读到
        #   文档A 的最后一个 chunk —— 而文档B 的 query 又"合法地"看得见自己的第一个
        #   chunk ⇒ 泄漏**绕过了所有 attention 掩码**（这就是为什么只修掩码不够）。
        #   实测：改文档A 的输入 ⇒ 文档B 输出变化 1.676e+00（掩码已全部正确）。
        #   ★ 为什么不能"把非本段的 chunk 清零再整体卷积"：卷积核仍会把那些零乘上权重、
        #     再和本段的值混在一起 ⇒ 结果依然变（只是幅度小了），不是精确解。
        #   做法：**逐段卷积** —— 每篇文档只把自己那段 chunk 喂给 conv。
        #   ★ 段数上限 = 每篇样本的文档数（Tc 个 chunk 最多 Tc 篇文档），
        #     一个 2048 序列通常几十篇 ⇒ Python 循环次数很少，代价可忽略。
        if single:
            self._dbg_single_calls = getattr(self, '_dbg_single_calls', 0) + 1  # ★插桩
            # 单文档：固定切分与分段切分完全等价（池化相同 + 卷积因果）⇒ 走快路径，
            # 保证「V3 不退化（逐位一致）」。判据由调用方算好 `single`（n_chunk==1）。
            hc = hc.transpose(1, 2)                               # (B, L, Tc)
            hc = self._conv(hc, Tc)
            hc = hc.transpose(1, 2)                               # (B, Tc, L)
            return self.latent_norm(hc)
        cid_c = cid.clamp(max=Tc - 1).to(torch.long)              # (B,T) 0..Tc-1
        out = hc.new_zeros(B, Tc, L)
        for b in range(B):
            # ★ 提速优化 2026-10-19（V100 提速③b）：chunk→文档映射一次
            #   scatter_reduce(amax) 建成（无效 token seg=-1 不影响 amax），再
            #   一次 tolist 在 Python 分组 —— 旧代码每样本 vb.any()+两次 gather、
            #   每文档 int(s)+nonzero 共约 15 次 D2H（40 步实测 docs_loop=16.5 万）。
            #   分组语义与旧代码**逐位相同**：同文档全部 chunk 收进同一 conv，
            #   收集顺序=下标升序（enumerate 即升序）。2026-10-16 的 -1 初始化
            #   语义保留：未占用 chunk 不归属任何文档、不进任何卷积窗。
            seg_b = seg[b].to(torch.long)                         # (T,) 无效位=-1
            cs = torch.full((Tc,), -1, dtype=torch.long, device=h.device)
            cs.scatter_reduce_(0, cid_c[b], seg_b, reduce='amax', include_self=True)
            groups = {}
            for ci, d in enumerate(cs.tolist()):                  # 1 次 D2H/样本
                if d >= 0:
                    groups.setdefault(d, []).append(ci)
            self._dbg_docs = getattr(self, '_dbg_docs', 0) + len(groups)  # ★插桩
            for idx_list in groups.values():
                m = len(idx_list)                                 # 该文档的 chunk 数
                # ★ 只把自己那组的 chunk **收集**出来再卷（不要去切前缀）：
                #   踩过的坑——写成 `hc[b, :idx.max()+1]` 时，后一篇文档的卷积窗口
                #   仍会读到前一篇文档的 chunk（前缀里含别人的数据），泄漏原样存在
                #   （实测：改文档A ⇒ 文档B 输出变化 1.676e+00）。
                idx = torch.tensor(idx_list, dtype=torch.long, device=h.device)
                sub = hc[b].index_select(0, idx).transpose(0, 1).unsqueeze(0)   # (1, L, m)
                sub = self._conv(sub, m).squeeze(0)               # (L, m)
                out[b, idx] = sub.transpose(0, 1)                 # 回填该文档的 chunk
        return self.latent_norm(out)

    def _conv(self, x: torch.Tensor, n_out: int) -> torch.Tensor:
        """压缩维度上的因果深度可分离卷积。

        ★★ 2026-10-16 修（P0-A 的第二个泄漏源）★★
          原代码是 `nn.Conv1d(..., padding=k-1)`（**双向**填充）+ `[:, :, :n_out]`，
          切出左侧 n_out 个位置后，位置 j 的卷积窗实际是**真实输入 j-(k-1)..j+(k-1)**
          ⇒ **含未来 k-1 个 chunk**。后果：压缩序列的**最后 k-1 个 chunk** 会看见
          "未来 chunk"的输入，打包训练时表现为「后一篇文档开头的 chunk 被前一篇文档
          污染」，且**换任何 attention 掩码都修不掉**。
          实测（最小复现）：把 chunk 0..7 的输入全换掉，chunk 8 的输出变化 1.212e+00、
          chunk 9 变化 1.244e+00 —— 而它们自己的输入一个字都没动。
          修法：模块 padding 设为 0，这里显式**只左填充 k-1**
          ⇒ 位置 j 只依赖 j-k+1..j（真因果）。
          `cfg.csa_causal_conv=False` 可退回旧行为（仅供 A/B 对照）。
        """
        k = self.conv.kernel_size[0]
        if k > 1:
            if getattr(self.cfg, 'csa_causal_conv', True):
                x = torch.nn.functional.pad(x, (k - 1, 0))     # 因果：只补左侧
            else:
                x = torch.nn.functional.pad(x, (k - 1, k - 1))  # 旧行为：双向（A/B 用）
        return self.conv(x)[:, :, :n_out]


    def forward(self, x: torch.Tensor, cache: Optional[Dict] = None,
                use_cache: bool = False, return_attn_max: bool = False,
                seg_mask: Optional[torch.Tensor] = None,
                seg_id: Optional[torch.Tensor] = None):
        """seg_mask: (B, 1, T, T) 或 (B, T, T) bool —— True = 可见（块对角，跨文档互不可见）
        seg_id:   (B, T) long —— 每个 token 的文档号（<0 = padding），分段池化用

        ★★ 2026-10-16 修（P0-A，实测确认的严重正确性缺陷）★★
          原签名**没有 seg_mask**，而 `TransformerBlockV3.attn_part` 只把 seg_mask 传给
          GlobalAttention（`isinstance(self.attn, GlobalAttention)`）⇒ 16 层里 12 层 CSA
          **完全没有跨文档掩码**。实测泄漏量：T=64、文档A/B 各 32 token、c=4 时，
          文档B 的 query 在「含文档A token 的 chunk」上的注意力权重合计 = 84.1466
          （128 个 (query,head) 对满权重为 128 ⇒ 约 66% 的注意力质量落在前一篇文档上）；
          反事实对拍：改文档A 的输入 ⇒ 文档B 位置输出变化 1.092e+00。
          修法：与 GlobalAttention **同语义**（块对角：同 seg 内因果可见，跨 seg 不可见），
          并把同一条掩码同时应用到【全局压缩分支】与【局部精确窗分支】——
          后者修的是反向泄漏（前一篇文档能看到后一篇的局部窗）。

        ★ 为什么分段池化必须显式要 seg_id（试过从 seg_mask 反推，**数学上不可能**）：
          块对角掩码只给出"两个 token 是否同文档"。文档A 的 token 3 与 token 31 掩码
          完全不同，但**没有任何依据判断 31 是文档末尾**；同理文档B 的 token 32 与 33
          掩码分别是 [32] 与 [32,33]，也无法知道 32 是起点。
          实测：用「可见 key 下标的最大值 − 最小值」当文档号，文档B 全体被判成同一个
          错误的值，chunk 划分失效，泄漏原样保留（判据③ 仍为 1.092e+00）。
          ⇒ 由 `TGAIV3Model.forward` 把真实 seg_id 直接传下来（它手上就有）。
        """
        B, T, _ = x.shape
        c = self.c
        use_seg = (seg_mask is not None) and getattr(self.cfg, 'csa_seg_isolate', True)
        Tc_fixed = (T + c - 1) // c

        seg = None
        cid = n_chunk = pos = None
        if use_seg:
            m = seg_mask
            if m.dim() == 4:
                m = m[:, 0]
            if m.dim() != 3 or m.shape[0] != B or m.shape[1] != T:
                use_seg = False
        if use_seg:
            if seg_id is not None:
                _s = seg_id.to(torch.int32)
                if _s.dim() == 1:
                    _s = _s.unsqueeze(0)
                if tuple(_s.shape) == (B, T):
                    seg = _s
            if seg is None:
                # 退化路径：只做「文档内/跨文档」的粗判 —— 用掩码把**每篇文档**标成
                # 一个块（同一文档的所有行共享同一可见起点集合的最小值）。
                # 精度不如真实 seg_id，但绝不比"完全不分段"更差。
                idx = torch.arange(T, device=x.device).view(1, 1, T)
                mi = m.to(torch.int32)
                row_any = m.any(dim=-1)
                mn = (mi * idx).masked_fill(~m, T).amin(dim=-1)
                seg = torch.where(row_any, mn, torch.full_like(mn, -1)).int()
            valid = (seg >= 0)
            # ★★ chunk 划分规则（两个条件缺一不可）★★
            #   ① 固定网格：每 c 个 token 一个新 chunk（t % c == 0）
            #   ② 文档边界：seg 与前一 token 不同 ⇒ 新 chunk
            #   ⇒ 每篇文档内部按 c 对齐切分，跨文档处强制切开。
            #   ⚠️ 踩过的两个坑（都会让 chunk 数彻底错）：
            #     · 写成 `cumsum(边界标志) - 1` ⇒ 首 token 被当成独立 chunk；
            #     · 只判 seg 变化 ⇒ 单文档时整段塌成 1 个 chunk（实测 _compress 差 3.9e+00）；
            #     · 漏了「t % c == 0」⇒ 每个 token 一个 chunk（同上）。
            tc = torch.arange(T, device=x.device)
            bnd = ((tc % c) == 0).unsqueeze(0).expand(B, T).to(torch.int64)
            if T > 1:
                bnd[:, 1:] = bnd[:, 1:] | (seg[:, 1:] != seg[:, :-1]).to(torch.int64)
            bnd[:, 0] = 1                       # 位置 0 永远是第一个 chunk 的起点
            cid_raw = torch.cumsum(bnd, dim=1) - 1              # (B,T) 0..
            cid = torch.where(valid, cid_raw, torch.zeros_like(cid_raw))
            # 无效（pad）位置不参与 chunk 计数
            n_chunk = torch.where(valid, cid_raw + 1,
                                  torch.zeros_like(cid_raw)).amax(dim=1)   # (B,)
            # ★ pos 就是 cid_raw（= 段内 chunk 序号 0,1,2..）：_compress 用它把 chunk
            #   按段内序号紧凑分组。不要另算 token 级序号 —— 实测那样会算成
            #   [0,0,0,0,1,2,3,4]（token 序号）而不是 [0,0,0,0,0,0,0,0]（chunk 序号）。
            pos = cid_raw
            # ★ 硬约束：分段切分出来的 chunk 数可能比固定切分**多**一点点
            #   （文档长度 l 需要 ceil(l/c) 个 chunk，而固定网格只给它 floor 的份额；
            #     只有 l % c != 0 时才多）。压缩后长度必须是 Tc = ceil(T/c)，
            #   否则 k/v / conv / 与推理路径的形状全都不一致。
            #   ⚠️ 不能用「把序号夹到 Tc-1」了事 —— 那会把两篇文档的 chunk 合并，
            #      隔离直接破掉（比原 bug 更糟）。这里选择**退化**：整个 batch 回到
            #      固定切分 + 纯注意力掩码。此时跨文档 chunk 虽然存在，但它的 cseg
            #      只会取到先出现的那篇文档的 seg ⇒ 后一篇文档的 query 看不见它，
            #      **不会泄漏**（代价：后一篇文档开头的 token 看不到自己那半截 chunk）。
            #     实测：文档长度是 c 整数倍时不触发（常见情形）。
            # ★ 提速优化 2026-10-19（V100 提速③c）：max().item() 2 次 D2H → 1 次
            _nc_max = int(n_chunk.max().item())
            _single = (_nc_max <= 1)
            if _nc_max > Tc_fixed:
                use_seg = False
                seg = cid = n_chunk = pos = None
                self._seg_fallback = getattr(self, '_seg_fallback', 0) + 1

        # ── 全局分支（对压缩序列）──
        kv = self.kv_down(x)                          # (B, T, latent)
        if use_seg:
            self._dbg_seg_calls = getattr(self, '_dbg_seg_calls', 0) + 1   # ★插桩
            kv_c = self._compress(kv, seg=seg, cid=cid, n_chunk=n_chunk, pos=pos,
                                  single=_single)
        else:
            self._dbg_fixed_calls = getattr(self, '_dbg_fixed_calls', 0) + 1  # ★插桩
            kv_c = self._compress(kv)                 # (B, T/c, latent)
        k = self.k_down(kv_c).view(B, -1, self.n_kv, self.d_k).transpose(1, 2)
        v = self.v_down(kv_c).view(B, -1, self.n_kv, self.d_k).transpose(1, 2)
        q = self.q_proj(x).view(B, T, self.n_q, self.d_k).transpose(1, 2)

        if self.cfg.attn_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        q_g = self.g_rope(q)
        k_g = self.g_rope(k)
        if self.n_kv != self.n_q:
            rep = self.n_q // self.n_kv
            k_g = k_g.repeat_interleave(rep, dim=1)
            v_g = v.repeat_interleave(rep, dim=1)
        else:
            v_g = v

        # 因果掩码：压缩位置 j 覆盖原始 token [j*c, (j+1)*c)，token i 可见 j <= i//c
        Tc = k_g.shape[2]
        if use_seg:
            # ★ 分段池化后 chunk 与 token 的对应关系不再是 j == i//c，改为查表：
            #   · cseg[b, j]     = chunk j 所属文档号（无效 chunk = -1 ⇒ 任何 query 都看不见）
            #   · tok_chunk[b, i]= token i 落在哪个 chunk
            #   可见条件 = 同段 且 chunk 序号 <= query 的 chunk 序号（因果）
            ar = torch.arange(Tc, device=x.device)
            cseg = torch.full((B, Tc), -1, dtype=seg.dtype, device=x.device)
            # ★ 用 scatter 把「每个 chunk 的文档号」填进 cseg：
            #   token i（batch b）落在 chunk cid[b,i] ⇒ cseg[b, cid[b,i]] = seg[b,i]
            #   · 无效（pad）位置与越界 chunk 一律写到 dummy 槽（cseg 的最后一格，
            #     因为我们保证 n_chunk <= Tc，所以 cid 真正的最大值 <= Tc-1；
            #     但 pad 位置的 cid 被置 0 会污染 chunk 0，必须排除）
            #   · flat 下标 = b*Tc + cid ⇒ 天然按 batch 隔离（踩过：忘了乘 Tc 会让
            #     后一个样本的 seg 覆盖前一个样本的 cseg）
            tok_chunk = cid                                  # (B, T)
            flat_cid = tok_chunk.reshape(-1).to(torch.int64)
            bidx = torch.arange(B, device=x.device).repeat_interleave(T)   # (B*T,)
            take = bidx * Tc + flat_cid
            dummy = B * Tc                                   # cseg 末尾追加的垃圾桶
            ok_tok = (seg.reshape(-1) >= 0)
            take_ok = torch.where(ok_tok, take, torch.full_like(take, dummy))
            take_ok = take_ok.clamp(0, dummy)
            cseg_ext = torch.cat(
                [cseg.reshape(-1), torch.full((1,), -1, dtype=seg.dtype, device=x.device)])
            cseg_ext.scatter_(0, take_ok, seg.reshape(-1))
            cseg = cseg_ext[:B * Tc].view(B, Tc)
            # ★ 有效 chunk = 「被至少一个有效 target token 占用」∧「它的 seg >= 0」。
            #   未被占用的 chunk 是补出来的零块（seg=-1），必须屏蔽，
            #   否则它们会以"全零 K/V"参与 softmax，把注意力质量吸走。
            occ = torch.zeros(dummy + 1, dtype=torch.float32, device=x.device)
            occ.scatter_add_(0, take_ok, ok_tok.to(torch.float32))
            valid_chunk = (occ[:B * Tc] > 0).view(B, Tc) & (cseg >= 0)
            cmask = ((cseg.unsqueeze(1) == seg.unsqueeze(-1)) &
                     valid_chunk.unsqueeze(1) &
                     (ar.view(1, 1, Tc) <= tok_chunk.unsqueeze(-1)))   # (B, T, Tc)
        else:
            idx_q = torch.arange(T, device=x.device) // c            # (T,)
            idx_k = torch.arange(Tc, device=x.device)                # (Tc,)
            cmask = (idx_k.unsqueeze(0) <= idx_q.unsqueeze(1))       # (T, Tc)
            cmask = cmask.unsqueeze(0).expand(B, -1, -1)             # (B, T, Tc)
        # ★ 统一成 (B, 1, T, Tc)：SDPA 的 batch 维不会自动广播，
        #   必须是 4D 且 batch 维 == q 的 batch 维（实测踩过：
        #   (1,T,Tc) 会报 "size of tensor a (4) must match tensor b (2) at dim 1"）
        attn_mask = torch.zeros(B, 1, T, cmask.shape[-1], device=x.device,
                                dtype=q_g.dtype)
        attn_mask.masked_fill_(~cmask.unsqueeze(1), float('-inf'))
        out_g, hm_g = _sdpa_with_max(q_g, k_g, v_g, attn_mask, return_attn_max)

        # ── 局部分支（精确窗口，保近邻交互）──
        w = min(self.local_win, T)
        k_l = k[:, :, -w:, :]
        v_l = v[:, :, -w:, :]
        q_l = self.l_rope(q)
        k_l = self.l_rope(k_l, offset=max(0, T - w))
        if self.n_kv != self.n_q:
            rep = self.n_q // self.n_kv
            k_l = k_l.repeat_interleave(rep, dim=1)
            v_l = v_l.repeat_interleave(rep, dim=1)
        lq = q_l.shape[2]
        lk = k_l.shape[2]
        if use_seg and lk > 0:
            # ★ 局部窗也要块对角 —— 修的是**反向泄漏**：
            #   窗口从右往左开，前一篇文档的末尾 query 会看到后一篇文档开头的 key。
            #   ★ key 是**压缩后的 chunk**（k 的序列维 = Tc，不是 T）：
            #     k_l = k[:, :, -w:, :] ⇒ 最后 lk = min(w, Tc) 个 chunk，key 下标 =
            #     Tc-lk .. Tc-1，与全局分支共用同一张 cseg / valid_chunk 表。
            cl = cseg[:, Tc - lk:]                                   # (B, lk)
            cl_ok = valid_chunk[:, Tc - lk:]                         # (B, lk)
            key_chunk = torch.arange(Tc - lk, Tc, device=x.device)   # (lk,)
            caus = key_chunk.view(1, 1, lk) <= tok_chunk.unsqueeze(-1)   # (B, lq, lk)
            same = ((seg.unsqueeze(-1) == cl.unsqueeze(1)) &
                    (seg >= 0).unsqueeze(-1) & cl_ok.unsqueeze(1))   # (B, lq, lk)
            same = same & caus
            lmask = torch.zeros(B, 1, lq, lk, device=x.device, dtype=q_l.dtype)
            lmask.masked_fill_(~same.unsqueeze(1), float('-inf'))
        else:
            # 无 seg_mask（推理/引擎路径）：标准因果掩码，3D/4D 都可，
            # 这里保持原来的 2D (lq, lk) —— SDPA 会广播到 (B,H,lq,lk)
            lmask = torch.full((lq, lk), float('-inf'), device=x.device, dtype=q_l.dtype)
            lmask = torch.triu(lmask, diagonal=lk - lq + 1)
        out_l, hm_l = _sdpa_with_max(q_l, k_l, v_l, lmask, return_attn_max)

        out = out_g + torch.sigmoid(self.gate) * out_l
        out = out.transpose(1, 2).reshape(B, T, self.n_q * self.d_k)

        # QK-Clip 取两个分支中较大的 head max（两个分支的 Q/K 是同一组权重）
        head_max = None
        if hm_g is not None or hm_l is not None:
            if hm_g is not None and hm_l is not None:
                head_max = torch.maximum(hm_g, hm_l)
            else:
                head_max = hm_g if hm_g is not None else hm_l
        return self.o_proj(out), head_max


# ═══════════════════════════════════════════════════════════
# MoE（升级：共享专家 + aux-loss-free 均衡 + z-loss）
# ═══════════════════════════════════════════════════════════
class MoEv3(nn.Module):
    """V3 MoE —— 严格按 DeepSeekMoE(arXiv:2401.06066) + DeepSeek-V3(arXiv:2412.19437)。

    论文公式（核实于 2026-10-14，见 docs/论文核实记录.md §2）：
        s_{i,t} = Softmax_i( u_t^T · e_i^l )                      ← **全专家 softmax**
        g_{i,t} = s_{i,t}  若 (s_{i,t}+b_i) ∈ Topk({s_{j,t}+b_j}, K_r)  否则 0
        h_t     = Σ_{i=1}^{K_s} FFN_i(u_t)                        ← 共享专家**直接相加**
                + Σ_{i>K_s} g_{i,t} FFN_i(u_t) + u_t

    ★ 三个关键点（初版写错，已按论文修正）：
      ① **偏置 b_i 仅用于 top-k 选择**；门控权重用【原始亲和度 s】，不是 (s+b)
      ② **偏置符号式更新**：过载则 b←b−γ，欠载则 b←b+γ（原文表述，非连续比例）
      ③ **共享专家直接相加，无门控**（sigmoid 门控是 Qwen 的做法，DeepSeek 不用）
    """

    def __init__(self, config: TGAIV3Config):
        super().__init__()
        self.cfg = config
        self.n_experts = config.n_experts
        self.top_k = config.n_activated
        self.gamma = config.moe_bias_lr          # ★ 偏置更新速度 γ
        self.z_loss_w = config.moe_router_z_loss
        self.alpha = config.moe_aux_alpha        # 互补序列辅助损失权重（V3 里很小）

        # 专家质心（router 权重即 e_i^l）
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        EXP = SiTUGLUFFN if config.moe_use_situ else SwiGLUFFN
        self.experts = nn.ModuleList([
            EXP(config.d_model, config.d_ff) for _ in range(config.n_experts)
        ])
        # ★ aux-loss-free / QB：偏置 buffer（不参与梯度，float32）
        self.register_buffer('expert_bias', torch.zeros(config.n_experts,
                                                        dtype=torch.float32))

        self.n_shared = config.n_shared_experts
        if self.n_shared > 0:
            self.shared_experts = nn.ModuleList([
                EXP(config.d_model, config.d_ff) for _ in range(self.n_shared)
            ])
        # ★ K3：routed 聚合后 RMSNorm（升维/输出前），稳定且正收益
        self.post_norm = (RMSNorm(config.d_model, config.norm_eps)
                          if config.moe_post_norm else None)

        self._z_loss = None
        self._aux_bal = None
        self._usage = None            # 本步每专家被选中的比例（用于偏置更新）
        self._ema_usage = None
        self._scores = None           # (N, E) sigmoid 分数（QB 用）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        N = xf.shape[0]

        # ① 全专家亲和度
        if self.cfg.moe_balance == 'quantile':
            # ★ K3 QB：s_i = Sigmoid(W_r x)（K3 Eq.13）
            s_all = torch.sigmoid(self.router(xf).float()).to(xf.dtype)
        else:
            # DeepSeek：Softmax over ALL experts
            s_all = F.softmax(self.router(xf).float(), dim=-1).to(xf.dtype)

        # ② top-k 选择：用 (s + b)，b 仅影响选择
        sel = s_all + self.expert_bias.to(s_all.dtype)
        topv, topi = torch.topk(sel, self.top_k, dim=-1)             # (N, K)

        # ③ ★ 门控权重
        if self.cfg.moe_balance == 'quantile':
            # K3：p_ij = s_ij / Σ_{r∈T} s_ir（在选中集合内归一化）
            gsel = torch.gather(s_all, 1, topi)
            gate = gsel / gsel.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        else:
            # DeepSeek：门控用原始亲和度（不是重新归一化）
            gate = torch.gather(s_all, 1, topi)

        # 统计负载（偏置更新用）
        with torch.no_grad():
            onehot = F.one_hot(topi, self.n_experts).float().sum(dim=1)   # (N, E)
            self._usage = onehot.mean(dim=0)                              # (E,)
            self._ema_usage = (self._usage if self._ema_usage is None
                               else 0.99 * self._ema_usage + 0.01 * self._usage)
            self._scores = s_all.detach()

        # ④ 按专家分组批算
        ids_flat = topi.reshape(-1)
        gate_flat = gate.reshape(-1)
        x_dup = xf.repeat_interleave(self.top_k, dim=0)
        # ★★ 2026-10-16 新增（P1-D）：两种实现，开关 `cfg.moe_impl` ★★
        #   'mask'  （默认，= 修复前行为）：布尔索引逐专家取子集
        #          `out_flat[m] = experts[eid](x_dup[m]) * gate[m]`
        #   'sorted'：一次 argsort 把 token 按专家排好，各专家在排序后是**连续切片**
        #          ⇒ 免掉 16 次布尔掩码的散射读写，实测单层 MoE fwd+bwd
        #          （N=8192）0.1425 s → 0.1113 s（1.28×，加速方案 §2.3）。
        #   ★ 两者数学等价（所以 A/B 必须 loss 曲线一致），默认仍为 mask。
        if getattr(self.cfg, 'moe_impl', 'mask') == 'sorted':
            sorter = torch.argsort(ids_flat, stable=True)
            x_sorted = x_dup.index_select(0, sorter)
            g_sorted = gate_flat.index_select(0, sorter)
            counts = torch.bincount(ids_flat, minlength=self.n_experts)
            bounds = torch.cumsum(counts, dim=0).tolist()
            sorted_out = torch.empty_like(x_sorted)
            start = 0
            for eid in range(self.n_experts):
                end = bounds[eid]
                if end > start:
                    seg = x_sorted[start:end]
                    # ★ 2026-10-18 根因修复：pad 到 8 的倍数
                    seg_p, _n = _pad_to_multiple(seg, 8)
                    _e = self.experts[eid](seg_p)[:_n]
                    sorted_out[start:end] = (_e
                                             * g_sorted[start:end].unsqueeze(-1))
                start = end
            # ★ 散回原 token 位置：`index_copy_` 的语义是 `self[index[i]] = source[i]`
            #   ⇒ 索引要传 **sorter**（不是它的逆置换 perm）。
            #   实测踩过：传逆置换时 19/20 行都错（max|Δ|=9.797e+00）。
            out_flat = torch.empty_like(sorted_out).index_copy(0, sorter, sorted_out)
        else:
            out_flat = torch.zeros_like(x_dup)
            # ★ 提速优化 2026-10-19（V100 提速①）：同步收敛。
            #   旧代码每专家 4 次 D2H 同步（.any() / x_dup[m] / gate_flat[m] / 赋值），
            #   16 专家 × 16 层 × 48 micro × 2(重算) ≈ 每步 10 万次、约 20 秒。
            #   现改为：bincount 一次取计数（1 次 D2H/层）替代逐专家 .any() 判断；
            #   x 与 gate 合并后单次布尔 gather（1 次）替代两次 gather。
            #   数学逐位不变（gate 与 x_dup 恒同 dtype：s_all 已 .to(xf.dtype)；
            #   若未来 gate 再变 fp32，cat 会提升精度、末尾 .to 兜底，行为同旧代码）。
            counts = torch.bincount(ids_flat, minlength=self.n_experts).tolist()
            xg = torch.cat([x_dup, gate_flat.unsqueeze(-1)], dim=1)
            for eid in range(self.n_experts):
                if counts[eid] == 0:
                    continue
                m = ids_flat == eid
                # ★ 2026-10-16 dtype 修复保留（见上）：赋值前 .to(out_flat.dtype)。
                # ★ 2026-10-18 根因修复：pad 到 8 的倍数，避免 cuBLAS fp16 未对齐 kernel
                sel = xg[m]                                    # 1 次同步
                # contiguous：sel 是跨步视图，避免 Linear 内部反复隐式拷贝
                _xs, _g = sel[:, :D].contiguous(), sel[:, D:]
                _xs_p, _n = _pad_to_multiple(_xs, 8)
                _e = self.experts[eid](_xs_p)[:_n]
                out_flat[m] = (_e * _g).to(out_flat.dtype)      # 1 次同步
        out = out_flat.view(N, self.top_k, D).sum(dim=1)

        # ⑤ ★ K3：routed 聚合后 RMSNorm（升维前）
        if self.post_norm is not None:
            out = self.post_norm(out)

        # ⑥ 共享专家：直接相加（DeepSeekMoE 论文）
        for se in self.shared_experts:
            out = out + se(xf)

        # ⑦ 辅助项（仅训练时记录）
        if self.training:
            logits = self.router(xf)
            z = torch.logsumexp(logits.float(), dim=-1) ** 2
            self._z_loss = z.mean()
            f = (self.n_experts / (self.top_k * N)) * onehot.sum(dim=0)
            P = s_all.mean(dim=0)
            self._aux_bal = (f * P).sum()
        return out.view(B, T, D)

    @torch.no_grad()
    def update_bias(self):
        """负载均衡偏置更新。

        两种模式（都满足"b 不进主梯度、只影响选择"）：
        ── 'quantile'（★ 默认，K3 Quantile Balancing，无超参）──
           论文（Kimi K3 Eq.14）：b̂_j ← −quantile_{1−k/n}(s_{:,j} − α)
              · α 是 token 侧门槛，用 Top-(k+1) 的第 k+1 名（省一次分位数）
              · 然后 b ← b̂ − mean(b̂)（中心化，避免整体漂移）
              · ★ 因果性：本 step 用旧 b 路由，新 b 下一步生效；推理时冻结
           价值：**去掉 γ 超参**（aux-free 的 γ 小则收敛慢、大则振荡）
        ── 'sign'（DeepSeek aux-loss-free）──
           原文："过载则偏置减少 γ，欠载则增大 γ"
        """
        if self._usage is None or self._scores is None:
            return
        n_exp = self.n_experts
        k = self.top_k

        if self.cfg.moe_balance == 'quantile':
            s = self._scores.float()                       # (N, E)
            # token 侧门槛 α_i：用带偏置的选择里第 k+1 名（若 k+1 > E 则取最小）
            sel = s + self.expert_bias.to(s.dtype)
            kk = min(k + 1, n_exp)
            alpha = torch.topk(sel, kk, dim=-1).values[:, -1]      # (N,)
            # r_ij = α_i − s_ij ; b̂_j = −quantile_{1−k/n}(r_{:,j})
            r = alpha.unsqueeze(1) - s                             # (N, E)
            q = 1.0 - k / n_exp
            b_hat = -torch.quantile(r, q, dim=0)                   # (E,)
            b_hat = b_hat - b_hat.mean()                           # 中心化
            self.expert_bias.copy_(b_hat.to(self.expert_bias.dtype))
        else:
            target = k / n_exp
            diff = self._usage - target
            self.expert_bias -= self.gamma * torch.sign(diff)
            self.expert_bias.clamp_(-1.0, 1.0)

    def aux_loss(self) -> torch.Tensor:
        """返回 z-loss 与互补序列辅助损失的加权和"""
        dev = self.router.weight.device
        z = (self._z_loss if self._z_loss is not None else torch.tensor(0.0, device=dev))
        b = (self._aux_bal if self._aux_bal is not None else torch.tensor(0.0, device=dev))
        return z * self.z_loss_w + b * self.alpha

    @property
    def usage(self):
        return self._usage


# ═══════════════════════════════════════════════════════════
# Transformer Block
# ═══════════════════════════════════════════════════════════
class TransformerBlockV3(nn.Module):
    """单个 Transformer 层。

    ★ Block AttnRes 需要在 **attn 之前** 和 **mlp 之前** 各做一次深度注意力，
      所以 forward 拆成 attn_part / mlp_part 两半，由主模型循环调度。
    """

    def __init__(self, config: TGAIV3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_global = config.is_global_layer(layer_idx)
        self.ln1 = RMSNorm(config.d_model, config.norm_eps)
        self.ln2 = RMSNorm(config.d_model, config.norm_eps)
        # 非全局层：启用 CSA → 压缩稀疏注意力；否则退化为标准因果注意力
        if self.is_global or not config.use_csa:
            self.attn = GlobalAttention(config, is_global=self.is_global)
            self.attn_kind = 'global' if self.is_global else 'local-causal'
        else:
            self.attn = CompressedSparseAttention(config)
            self.attn_kind = 'csa'
        self.ffn = (SwiGLUFFN(config.d_model, config.d_ff) if config.dense_ffn
                    else MoEv3(config))
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def attn_part(self, h: torch.Tensor, cache: Optional[Dict] = None,
                  use_cache: bool = False, return_attn_max: bool = False,
                  seg_mask: Optional[torch.Tensor] = None,
                  seg_id: Optional[torch.Tensor] = None):
        # ★ 2026-10-16 修：原来无条件传 seg_mask，而 CompressedSparseAttention
        #   的 forward 没有这个参数 ⇒ csa_compress>1 的预设（save_* 全系列）
        #   连一次 forward 都跑不了。改为按类型过滤，两种注意力都能用。
        # ★★ 2026-10-16 再修（P0-A）：过滤掉 seg_mask 正是 12/16 层跨文档泄漏的根因。
        #   现在 CSA 也接受 seg_mask / seg_id（同语义：块对角 + 因果 + 分段池化），
        #   所以**两种注意力都传**；只有 csa_seg_isolate=False 时才退回旧的过滤行为。
        _kw = dict(cache=cache, use_cache=use_cache,
                   return_attn_max=return_attn_max)
        if isinstance(self.attn, GlobalAttention):
            _kw['seg_mask'] = seg_mask
        elif getattr(self.attn.cfg, 'csa_seg_isolate', True):
            _kw['seg_mask'] = seg_mask
            _kw['seg_id'] = seg_id          # ★ 分段池化必须知道真实文档号
        o, hm = self.attn(self.ln1(h), **_kw)
        return self.drop(o), hm

    def mlp_part(self, h: torch.Tensor) -> torch.Tensor:
        return self.drop(self.ffn(self.ln2(h)))

    # 兼容旧调用（无 AttnRes 时）
    def forward(self, x: torch.Tensor, cache: Optional[Dict] = None,
                use_cache: bool = False):
        a, _ = self.attn_part(x, cache, use_cache)
        return x + a + self.mlp_part(x + a)


# ═══════════════════════════════════════════════════════════
# AttnRes（K3 Block Attention Residuals，arXiv:2603.15031）
# ═══════════════════════════════════════════════════════════
class BlockAttnRes(nn.Module):
    """Block Attention Residuals —— 严格按官方伪代码实现。

    论文（arXiv:2603.15031，Kimi Team, 2026）：
      标准残差是固定单位权重累加，深层会稀释每层贡献并让 PreNorm 下隐状态幅度无界增长。
      AttnRes 改为**对前层输出做 softmax 注意力**：
          h_l = Σ_{i<l} α_{i→l} · v_i
      权重 α 由**每层一个学习的伪查询 w_l ∈ R^d** 产生。
      Block 版：层分 N 个 block，block 内普通残差累加（partial_block），
      只对「已完成 block 表示 + 当前 partial sum」做注意力 ⇒ 内存 O(Nd)。

    官方伪代码（逐行照搬）：
        V = stack(blocks + [partial_block])                  # [N+1, B, T, D]
        K = norm(V)
        logits = einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)
        h = einsum('n b t, n b t d -> b t d', logits.softmax(0), V)

    实测收益（论文，Kimi Linear 48B/3B激活，1.4T tokens）：
        GPQA-Diamond 36.9→44.4 · HumanEval 59.1→62.2 · C-Eval 79.6→82.5
        Block AttnRes ≈ 基线用 1.25× 算力
    """

    def __init__(self, config: TGAIV3Config):
        super().__init__()
        # ★ 伪查询：长度 d 的向量（不是 Linear 层输出）
        self.proj = nn.Linear(config.d_model, 1, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        self.norm = RMSNorm(config.d_model, config.norm_eps)

    def forward(self, blocks: List[torch.Tensor],
                partial_block: torch.Tensor) -> torch.Tensor:
        if not blocks:
            return partial_block
        # ★★ 2026-10-16 修（父 agent 实测抓到的 OOM 热点）★★
        #   原写法：`V = stack(...)`, `K = self.norm(V)` —— 对**堆叠后的
        #   [N+1, B, T, D] 整体做 RMSNorm。RMSNorm 内部 `x.float()` 会产生一份
        #   fp32 大张量（B=2,T=2048,D=1024 时 268 MB），再加上必须保存到反向的
        #   K（又 268 MB，autocast 下 K 还是 fp32）⇒ 每个 block 多 ~0.5 GB，
        #   且 blocks 越多越贵（16 层/attn_res_block=2 ⇒ 最多 9 个），
        #   实测 B=2 就 OOM（报错点就是这里的 `x.float().pow(2)`）。
        #   ★ 修法：RMSNorm 是**逐位置、只沿最后一维**的运算 ⇒ 逐块归一化与
        #     "先 stack 再归一化" 数学完全等价（都是 x/rms(x)·γ），
        #     但不再产生整块 fp32 副本，反向也只需要各块的 K。
        #   注：K 用 no_grad 算（γ 需要梯度，故对 weight 单独保留梯度路径）。
        Ks, Vs = [], []
        for t in list(blocks) + [partial_block]:
            # 逐块 norm：γ 的梯度照常回传（RMSNorm.forward 内部乘 self.weight）
            Ks.append(self.norm(t))
            Vs.append(t)
        V = torch.stack(Vs, dim=0)                            # [N+1, B, T, D]
        K = torch.stack(Ks, dim=0)                            # [N+1, B, T, D]
        # 官方写法 einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)
        w = self.proj.weight.reshape(-1)                      # (D,)
        logits = torch.einsum('d,nbtd->nbt', w.to(K.dtype), K)
        h = torch.einsum('nbt,nbtd->btd', logits.softmax(0), V)
        return h


# ═══════════════════════════════════════════════════════════
# 主模型
# ═══════════════════════════════════════════════════════════
class TGAIV3Model(nn.Module):
    def __init__(self, config: TGAIV3Config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model,
                                  padding_idx=config.pad_token_id)
        self.blocks = nn.ModuleList([
            TransformerBlockV3(config, i) for i in range(config.n_layers)
        ])
        # ★ Block AttnRes：每个 attn / mlp 各有独立的 proj + norm（照官方）
        self.use_attn_res = config.attn_res_enabled and config.dense_attn_res_free
        if self.use_attn_res:
            self.attn_res_proj = nn.ModuleList(
                [BlockAttnRes(config) for _ in range(config.n_layers)])
            self.mlp_res_proj = nn.ModuleList(
                [BlockAttnRes(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # ★ QK-Clip 遥测：每层每头的 pre-softmax 最大分数（buffer，不进 checkpoint 优化器）
        self.register_buffer('_qk_max', torch.zeros(config.n_layers, config.n_q_heads),
                             persistent=False)

        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if self.config.pad_token_id is not None:
                with torch.no_grad():
                    m.weight[self.config.pad_token_id].zero_()

    # ── 前向 ──
    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                caches: Optional[List[Dict]] = None,
                use_cache: bool = False,
                return_qk_max: bool = False,
                seg_id: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x = self.embed(input_ids)
        cfg = self.config
        ckpt = self.gradient_checkpointing and self.training and targets is not None
        bs = max(1, cfg.attn_res_block)
        # ★★ P1-C：QK-Clip 遥测按间隔采样 ★★
        #   由 Trainer 在 forward 前设置 `self._qk_sample`（None = 每步都算）。
        #   非采样步直接不算 q@kᵀ（省 8.5% fwd+bwd 与一份 (B,H,T,T) fp32 临时显存）。
        if getattr(self, '_qk_sample', None) is False:
            return_qk_max = False

        # ★ 打包训练：由 seg_id 构造块对角掩码（跨文档互不可见）
        seg_mask = None
        if seg_id is not None:
            from data_v3 import make_seg_mask
            seg_mask = make_seg_mask(seg_id, device=input_ids.device)

        if self.use_attn_res:
            # ★ 严格照官方伪代码：
            #   blocks      = 已完成的 block 表示（初始为空，即第 0 层 attn 前只看到 embedding）
            #   partial     = 当前块内的累加和（含 embedding 起步）
            #   block 边界按 (i+1) % block 判定 —— 官方用 layer_number % (block_size//2)，
            #   block_size 计 ATTN+MLP（每层算 2），故每 block_size//2 层收一次，等价于我们的 bs
            blocks: List[torch.Tensor] = []
            partial = x
            qk_rows = []
            for i, blk in enumerate(self.blocks):
                # ① attn 前的深度注意力（blocks + 当前 partial）
                h = self.attn_res_proj[i](blocks, partial)
                if ckpt:
                    def _ap(hh, b=blk, ci=i, sm=seg_mask, si=seg_id):
                        return b.attn_part(hh, cache=None, use_cache=False,
                                           return_attn_max=return_qk_max,
                                           seg_mask=sm, seg_id=si)
                    a, hm = torch.utils.checkpoint.checkpoint(
                        _ap, h, use_reentrant=False, preserve_rng_state=False)
                else:
                    a, hm = blk.attn_part(
                        h, cache=(caches[i] if caches is not None else None),
                        use_cache=use_cache, return_attn_max=return_qk_max,
                        seg_mask=seg_mask, seg_id=seg_id)
                if hm is not None:
                    qk_rows.append((i, hm))
                # ② 块内累加（残差）
                new_partial = partial + a
                # ③ 到达 block 边界：把当前 partial 收进 blocks，开新块
                if (i + 1) % bs == 0:
                    blocks.append(new_partial)
                partial = new_partial
                # ④ mlp 前的深度注意力
                h = self.mlp_res_proj[i](blocks, partial)
                if ckpt:
                    m = torch.utils.checkpoint.checkpoint(
                        blk.mlp_part, h, use_reentrant=False, preserve_rng_state=False)
                else:
                    m = blk.mlp_part(h)
                partial = partial + m
            x = partial
        else:
            qk_rows = []
            for i, blk in enumerate(self.blocks):
                if ckpt:
                    def _fn(h, b=blk, ci=i, sm=seg_mask, si=seg_id):
                        a, hm = b.attn_part(
                            h, cache=None, use_cache=False, seg_mask=sm, seg_id=si,
                            return_attn_max=return_qk_max)
                        # ★★ 2026-10-16 修：grad_ckpt 下 QK 遥测恒为 0 ★★
                        #   原闭包写死 `return_attn_max=False` ⇒ 实测 B=1..4 + grad_ckpt=1
                        #   时 QK 非零层数 = 0，而同配置 grad_ckpt=0 时 = 16 层 / max=42.7。
                        #   grad_ckpt 默认开 ⇒ QK 监控（K2 报告的核心稳定性指标）完全失效。
                        #   修法：闭包内照常算 hm，**先收集到列表**，本轮前向结束后
                        #   由外层统一的 no_grad 循环写 buffer（与 ckpt=0 分支同一套写法）。
                        #   ⚠️ 不能在闭包内直接写 buffer：会让 checkpoint 的输出依赖 buffer，
                        #      报 "a leaf Variable that requires grad is being used in an
                        #      in-place operation"（本实现踩过）。
                        #   ⚠️ 也不要依赖 `is_recomputing()`：实测在 use_reentrant=False
                        #      的重算阶段仍为 False（本实现踩过，遥测依旧是 0）。
                        #   注意：重算阶段也会执行这里，会重复 append —— 无害（同样的值，
                        #   外层只写最后一次），但用 ci 去重更省。
                        if hm is not None and ci not in _seen:
                            qk_rows.append((ci, hm))
                            _seen.add(ci)
                        return h + a + b.mlp_part(h + a)
                    _seen = set()
                    x = torch.utils.checkpoint.checkpoint(_fn, x,
                                                          use_reentrant=False)
                else:
                    a, hm = blk.attn_part(
                        x, cache=(caches[i] if caches is not None else None),
                        use_cache=use_cache, return_attn_max=return_qk_max,
                        seg_mask=seg_mask, seg_id=seg_id)
                    if hm is not None:
                        qk_rows.append((i, hm))
                    x = x + a + blk.mlp_part(x + a)

        # QK-Clip 遥测写入 buffer
        if qk_rows:
            with torch.no_grad():
                H = self._qk_max.shape[1]
                for i, hm in qk_rows:
                    # ★★ 2026-10-16 修（父 agent 实测抓到的硬 bug）★★
                    #   原写法 `n = min(hm.numel(), H); self._qk_max[i,:n] = hm[:n]`
                    #   在 B>1 时炸：hm 若为 (B,H)，numel()=B*H ⇒ n=H ⇒ `hm[:H]`
                    #   是 (B,H)（B<H 时），赋给 (H,) 报
                    #   "expand(...{[2,16]}, size=[16])" —— microB≥2 完全不能用。
                    #   修法：统一按 batch 归约成 (H,)（与 _sdpa_with_max 出口一致），
                    #   并顺带处理 (B,H) / (H,) / 任意维度三种输入形状。
                    h = hm
                    while h.dim() > 1:
                        h = h.amax(dim=0)          # (B,H)→(H,) ; 更多维时逐维归约
                    n = min(h.numel(), H)
                    if n > 0:
                        self._qk_max[i, :n] = h[:n].to(self._qk_max.dtype)

        x = self.final_norm(x)
        # ★★ 2026-10-16 重写（父 agent 实测定位）：分块必须切到 **lm_head 之前** ★★
        #   真正的显存大头是 `lm_head(x)` 输出的 logits：B×T×96K。
        #   实测（B=4,T=2048,96K）：fp16 logits 1.5 GB，**反向还要再存一份 fp32 = 3.0 GB**，
        #   再加上 `logits[:, :-1].contiguous()` 又复制一份 ⇒ 单个 CE 步要 ~6 GB。
        #   原实现把 logits 整个算完再分块算 CE，**根本没省到 logits 本身**
        #   ⇒ 实测 B=1 关梯度检查点就吃 21 GB，B=3 直接 OOM。
        #   现在：先把 x 切成 chunk，**每个 chunk 单独过 lm_head + CE**，
        #   峰值只保留 1 个 chunk 的 logits（省 (B×T/chunk) 倍）。
        #   ★ 数学等价：CE 的 shift 就是在 x/labels 上各错一位，按 chunk 切不影响。
        #   ⚠️ logits 的完整张量仅在 cch<=0（旧行为）或需要返回时构造。
        cch = int(getattr(cfg, 'ce_chunk', 0) or 0)
        loss = None
        if targets is not None and cch > 0:
            # shift 一位：用前 t 个 token 预测第 t+1 个（与下面 cch=0 分支同语义）
            xc = x[:, :-1, :]                       # (B, T-1, D)
            lc = targets[:, 1:]                     # (B, T-1)
            SL = xc.size(1)
            nvalid = int((lc != -100).sum().item())
            tot = x.new_zeros(())
            for s0 in range(0, SL, cch):
                s1 = min(s0 + cch, SL)
                lg = self.lm_head(xc[:, s0:s1, :]).reshape(-1, self.config.vocab_size)
                tot = tot + F.cross_entropy(lg, lc[:, s0:s1].reshape(-1),
                                            ignore_index=-100, reduction='sum')
            loss = tot / max(1, nvalid)
            # 训练不需要完整 logits（省一份 (B,T,96K)）；推理 targets=None 时走 else 分支
            logits = None
        else:
            logits = self.lm_head(x)
            if targets is not None:
                # ★★ 关键：**错位一位**再做交叉熵（2026-10-15 实测抓到的严重 bug）★★
                #   打包数据的 labels[t] == input_ids[t]（同一位置），
                #   而 logits[t] 是由 input_ids[0..t] 算出的（已包含 ids[t] 本身）。
                #   若直接 cross_entropy(logits, targets)，模型就是在**抄输入**
                #   ⇒ loss 虚低到 0.6（正常应 2.5~4.5），训练看似正常实则无效。
                #   正确做法（与 HuggingFace/GPT 一致）：
                #       shift_logits = logits[:, :-1]   （位置 0..T-2）
                #       shift_labels = targets[:, 1:]   （位置 1..T-1）
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = targets[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100)
        return logits, loss, None

    # ── 辅助 loss（MoE z-loss）──
    def aux_loss(self) -> torch.Tensor:
        if self.config.dense_ffn:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        total = None
        for blk in self.blocks:
            if isinstance(blk.ffn, MoEv3):
                a = blk.ffn.aux_loss()
                a = a.to(next(self.parameters()).device)
                total = a if total is None else total + a
        return total if total is not None else torch.tensor(0.0)

    def update_moe_bias(self):
        """训练循环每步调用（aux-loss-free 均衡）。"""
        if self.config.dense_ffn:
            return
        for blk in self.blocks:
            if isinstance(blk.ffn, MoEv3):
                blk.ffn.update_bias()

    @torch.no_grad()
    def moe_stats(self) -> Dict:
        """返回各层专家使用率，用于诊断路由是否失衡。"""
        out = {}
        for i, blk in enumerate(self.blocks):
            if isinstance(blk.ffn, MoEv3) and blk.ffn._ema_usage is not None:
                out[i] = blk.ffn._ema_usage.tolist()
        return out

    @torch.no_grad()
    def qk_max_stats(self) -> Dict:
        """★ QK-Clip 诊断：各层各头的 pre-softmax 最大注意力分数。

        判据（Kimi K2 报告）：正常应在数十量级；若某些 head 远超 τ（默认 100），
        说明该 head 正在跑飞，需要 QK-Clip 干预或降低学习率。
        """
        m = self._qk_max
        return {'overall_max': float(m.max()),
                'per_layer_max': m.amax(dim=1).tolist(),
                'n_over_tau': int((m > self.config.qk_clip_tau).sum().item())
                if self.config.qk_clip_tau > 0 else 0}

    def enable_gradient_checkpointing(self, on: bool = True):
        self.gradient_checkpointing = on

    # ── 生成（带 KV Cache）──
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64,
                 temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9,
                 eos_token_id: int = EOS_ID, repetition_penalty: float = 1.0
                 ) -> torch.Tensor:
        self.eval()
        device = input_ids.device
        caches = [{} for _ in range(self.config.n_layers)]
        out = input_ids
        generated = 0
        for _ in range(max_new_tokens):
            cur = out if generated == 0 else out[:, -1:]
            logits, _, _ = self.forward(cur, caches=caches, use_cache=True,
                                        return_qk_max=False)
            nxt = logits[:, -1, :].float()
            if repetition_penalty and repetition_penalty != 1.0 and generated > 0:
                # ★ 2026-10-15 修正方向 bug：原版是**无条件** `/= penalty`，
                #   对负 logit 会让它绝对值变小 ⇒ 概率变大 ⇒ **反而鼓励重复**。
                #   CTRL 论文原式（正确）：
                #       score > 0 → score / penalty
                #       score < 0 → score * penalty
                for b in range(out.shape[0]):
                    uniq = torch.unique(out[b])
                    sel = nxt[b, uniq]
                    nxt[b, uniq] = torch.where(
                        sel > 0, sel / repetition_penalty,
                        sel * repetition_penalty)
            if temperature and temperature > 0:
                nxt = nxt / temperature
            if top_k and top_k > 0:
                v, _ = torch.topk(nxt, min(top_k, nxt.size(-1)))
                nxt[nxt < v[:, [-1]]] = float('-inf')
            probs = F.softmax(nxt, dim=-1)
            if top_p and 0 < top_p < 1:
                sp, si = torch.sort(probs, descending=True, dim=-1)
                cum = sp.cumsum(dim=-1)
                mask = cum - sp > top_p
                sp[mask] = 0.0
                sp = sp / sp.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                idx = torch.multinomial(sp, 1)
                nxt_tok = si.gather(-1, idx)
            else:
                nxt_tok = torch.multinomial(probs, 1)
            out = torch.cat([out, nxt_tok], dim=1)
            generated += 1
            if eos_token_id is not None and bool((nxt_tok == eos_token_id).all()):
                break
        return out

    # ── 参数量统计 ──
    def param_report(self) -> Dict[str, int]:
        emb = self.embed.weight.numel()
        head = 0 if self.config.tie_embeddings else self.lm_head.weight.numel()
        attn = sum(p.numel() for b in self.blocks
                   for p in b.attn.parameters())
        ffn = sum(p.numel() for b in self.blocks for p in b.ffn.parameters())
        norm = sum(p.numel() for n, p in self.named_parameters()
                   if 'norm' in n or 'ln' in n)
        ar = 0
        if self.use_attn_res:
            ar = (sum(p.numel() for p in self.attn_res_proj.parameters())
                  + sum(p.numel() for p in self.mlp_res_proj.parameters()))
        total = sum(p.numel() for p in self.parameters())
        # 激活参数（每 token）：只算 top-k 专家 + 共享专家 + 注意力
        act = emb + head + attn + ar + norm
        per_expert = 0
        if not self.config.dense_ffn:
            ex = self.blocks[0].ffn.experts[0]
            per_expert = sum(p.numel() for p in ex.parameters())
            act += per_expert * self.config.n_activated * self.config.n_layers
            if self.config.n_shared_experts > 0:
                act += sum(p.numel() for p in self.blocks[0].ffn.shared_experts.parameters()) \
                       * self.config.n_layers
        else:
            act += sum(p.numel() for p in self.blocks[0].ffn.parameters()) * self.config.n_layers
        return {'total': total, 'active': act, 'embed': emb, 'head': head,
                'attn': attn, 'ffn': ffn, 'norm': norm, 'attn_res': ar,
                'per_expert': per_expert}


# ═══════════════════════════════════════════════════════════
# 尺寸预设（一套代码换规模）
# ═══════════════════════════════════════════════════════════
def preset(name: str) -> TGAIV3Config:
    """按名字取预设配置。

    world 系列（ESP32-S3，词表 8K，稠密或小 MoE）
        'w10m'      ~10M
        'w22m'      ~22M（ESP32-S3 PSRAM 上限附近）
    SAVE 系列（词表 96K，MoE）
        'save_0_5b' / 'save_0_86b' / 'save_1_5b' / 'save_3b'
    'v3_7b'         论文档目标（20 层 / 8 专家激活 2）
    """
    k = name.lower()
    if k == 'w10m':
        # ESP32-S3 儿童款 —— 依据《ESP32-S3_10M模型_研究复命.md》(2026-10-14 联网实测)：
        #   · PSRAM 顺序读实测 58-61 MB/s ⇒ 10M/4bit 每 token 读 5.35MB ⇒ 带宽天花板 11.4 tok/s
        #     规划值 3-8 tok/s（目标 5）；真瓶颈是 int4 解包+标量计算，不是带宽
        #   · 配置照研究推荐的「配置 A」：d_model 256 / 12 层 / 8 Q头 / **1 KV头(MQA)**
        #     SwiGLU d_ff 672 / vocab 8192 **tied** / RoPE θ=10000
        #   · ★ θ 改 10000（研究：θ=500K 在训练长度内无收益，还把 1/3 位置维度退化成 NoPE）
        #   · ★ 砍压缩稀疏注意力（attention 只占 24% 算力；稀疏要 gather 不连续 KV，
        #     正踩 PSRAM 随机读慢 5-7× 的痛点）
        #   · ★ 保留 AttnRes（低成本）；层共享作为 A/B 实验项（layer_share_group=2 可开）
        #
        # ★★ 上下文长度（2026-10-15 用户明确）★★
        #   ESP32 端**实测 128 能跑，极限未知** ⇒ 首版训练就按 128，与部署一致。
        #   理由：研究实测「超出训练长度就崩」（θ=500K 在 L=1512 时 ppl 37.9 vs θ=512 的 21.2），
        #         所以**训练长度必须 = 部署长度**，不能训 1024 再压到 128 用。
        #   θ 也按训练长度取（研究表格里 θ=L_train=512 时外推最好 ⇒ θ≈L_train 是安全选择，
        #   但我们取 10000 的标准值，因为 ESP32 上不做外推）。
        #   ⇒ 后续用 `--seq-len` 扫参实测 ESP32 极限（128 / 192 / 256 / 384 / 512）
        return TGAIV3Config(
            vocab_size=8192, d_model=256, n_layers=12, n_heads=8, d_ff=672,
            max_seq_len=128,            # ★ ESP32 实测可用值
            # 每 4 层 1 层全局 ⇒ 12 层 = 9 局部 + 3 全局
            attn_global_every=4,
            n_kv_heads=1,               # ★ MQA（研究：KV 流量减半，MobileLLM 实测几乎无损）
            csa_compress=1,             # ★ =1 表示不做序列压缩（标准因果注意力）
            csa_latent=64, csa_local_window=128,
            attn_res_block=4,           # 12/4 = 3 个 block
            dense_ffn=True,             # 10M 保持稠密（研究建议 V2 稠密交付）
            n_experts=1, n_activated=1, n_shared_experts=0,
            rope_theta=10000.0,         # ★ 研究结论
            qk_clip_tau=0.0)            # 小模型不需 QK-Clip
    if k == 'w22m':
        # ESP32-S3 上限档（PSRAM 8MB 内，4bit 约 11MB 偏紧 ⇒ 建议 3bit）
        return TGAIV3Config(
            vocab_size=8192, d_model=384, n_layers=14, n_heads=12, d_ff=1024,
            max_seq_len=2048, attn_global_every=4,
            n_kv_heads=2, csa_compress=1, csa_latent=96, csa_local_window=192,
            attn_res_block=4, dense_ffn=True,
            n_experts=1, n_activated=1, n_shared_experts=0,
            rope_theta=10000.0, qk_clip_tau=0.0)
    if k == 'save_0_5b':
        # 10 层 × 1024 + 8 专家激活 2（0.5B 档）
        #   AttnRes block=5 ⇒ 10/5 = 2 个 block（层少，block 不宜太小）
        return TGAIV3Config(
            vocab_size=96000, d_model=1024, n_layers=10, n_heads=16, d_ff=1280,
            max_seq_len=8192, n_experts=8, n_activated=2, n_shared_experts=1,
            attn_global_every=4, n_kv_heads=4, csa_compress=4, csa_latent=256,
            csa_local_window=128, attn_res_block=5, rope_theta=500000.0)
    if k == 'save_0_86b':
        # ★ 首选基座：16 层 × 1024 + 8 专家激活 2（实测 875M / 激活 384M）
        #   AttnRes block=4 ⇒ 16/4 = 4 个 block（论文说约 8 个 block 恢复大部分收益，
        #   我们层数少，4 个 block 是合理折中）
        return TGAIV3Config(
            vocab_size=96000, d_model=1024, n_layers=16, n_heads=16, d_ff=1664,
            max_seq_len=8192, n_experts=8, n_activated=2, n_shared_experts=1,
            attn_global_every=4, n_kv_heads=4, csa_compress=4, csa_latent=256,
            csa_local_window=128, attn_res_block=4, rope_theta=500000.0)
    if k == 'save_1_5b':
        # ★ 0.86B 扩容目标之一：加宽 d_model/FFN，层数不变
        #   （更稳的扩容路径是只增专家：见 graft_experts 用法示例）
        return TGAIV3Config(
            vocab_size=96000, d_model=1280, n_layers=16, n_heads=20, d_ff=1536,
            max_seq_len=16384, n_experts=8, n_activated=2, n_shared_experts=1,
            attn_global_every=4, n_kv_heads=4, csa_compress=4, csa_latent=320,
            csa_local_window=128, attn_res_block=4, rope_theta=500000.0)
    if k in ('save_v3_1_5b', 'save_v3_1.5b', 'v3_1_5b'):
        # ★ 2026-10-16 新增：TGAI SAVE V3 (1.5B) 正式预设
        #   依据架构评审（见 V3 设计文档 §十二），与 save_1_5b_exp 的差异：
        #     · attn_res_block 4 -> 2   （16 层 = 8 个 block，对齐论文的 N≈8）
        #     · hetero_rope 开启        （双轨 θ：局部 10000 / 压缩 160000）
        #     · n_kv_heads 4 -> 1       （MQA，对齐 V4 生产配置）
        #     · sliding_window 128      （对齐 V4 的 sliding_attention）
        #     · max_seq_len 2048        （★ 用户要求：预训练 2K，SFT 再阶梯推到 4K/8K）
        #   参数预算（d=1024 / L=16 / 16 专家 / ff=1664）：
        #     embed 96Kx1024 = 98.3M（tied）
        #     attn  ~16 x 4.2M  = 67M
        #     MoE   16 x (16 x 3x1024x1664 + shared) ≈ 1.35B
        #     ⇒ 合计约 1.52B
        return TGAIV3Config(
            vocab_size=96000, d_model=1024, n_layers=16, n_heads=16, d_ff=1664,
            max_seq_len=2048,
            n_experts=16, n_activated=2, n_shared_experts=1,
            moe_balance='quantile',          # K3 QB，无 γ 超参
            moe_use_situ=True, moe_post_norm=True,
            attn_global_every=4,
            n_kv_heads=1,                    # ★ MQA（评审第③条）
            csa_compress=4, csa_latent=256, csa_local_window=128,
            sliding_window=128,              # ★ 评审第⑤条
            attn_res_block=2,                # ★ 评审第①条（16/2 = 8 block）
            attn_res_enabled=True,
            attn_qk_norm=True,
            qk_clip_tau=100.0, qk_clip_alpha=0.5,
            hetero_rope=True,                # ★ 评审第②条（双轨）
            rope_theta=10000.0,              # 基线（局部轨）
            rope_theta_local=10000.0,
            rope_theta_global=160000.0,      # 压缩轨（V4 用 160000 + YaRN16）
            partial_rotary_global=0.25,
            partial_rotary_local=1.0,
        )

    if k in ('save_1_5b_exp', 'save_1_5b_expand', 'save_1_5b_pure'):
        # ★ 2026-10-16 新增（用户要求："通过新增新专家的方式升级"）
        #   与 save_1_5b 的区别：**d_model 保持 1024 不变**，只把专家 8→16。
        #   为什么：graft_experts 硬性要求 d_model 相同（它只在同 d_model 下
        #   保留旧专家权重、复制新增专家、给路由新列负偏置）。
        #   原有 save_1_5b 是 d_model=1280 ⇒ 扩容路径断掉。
        #   参数量：98.3M(embed tied) + 16 层 x (4.2M attn + 16x5.11M MoE)
        #           ≈ 98.3 + 16 x 86.0 ≈ 1.474B（激活约 0.47B）
        return TGAIV3Config(
            vocab_size=96000, d_model=1024, n_layers=16, n_heads=16, d_ff=1664,
            max_seq_len=8192, n_experts=16, n_activated=2, n_shared_experts=1,
            attn_global_every=4, n_kv_heads=4, csa_compress=4, csa_latent=256,
            csa_local_window=128, attn_res_block=4, rope_theta=500000.0)

    if k == 'save_3b':
        # ★ 0.86B 扩容目标：专家池 8→16 + 层 16→24（嫁接而来）
        return TGAIV3Config(
            vocab_size=96000, d_model=1536, n_layers=24, n_heads=24, d_ff=1600,
            max_seq_len=32768, n_experts=16, n_activated=3, n_shared_experts=1,
            attn_global_every=4, n_kv_heads=6, csa_compress=4, csa_latent=384,
            csa_local_window=256, attn_res_block=4, rope_theta=500000.0)
    if k == 'v3_7b':
        # V3 文档目标：32 层 / d=2560 / 8 专家激活 2 / 64K
        #   AttnRes block=4 ⇒ 8 个 block（正好符合论文的"约 8 个 block"）
        return TGAIV3Config(
            vocab_size=96000, d_model=2560, n_layers=32, n_heads=20, d_ff=2560,
            max_seq_len=65536, n_experts=8, n_activated=2, n_shared_experts=1,
            attn_global_every=4, n_kv_heads=5, csa_compress=4, csa_latent=640,
            csa_local_window=512, attn_res_block=4, rope_theta=500000.0)
    raise ValueError('未知预设: %s' % name)


# ═══════════════════════════════════════════════════════════
# 专家扩容嫁接（graft）
# ═══════════════════════════════════════════════════════════
def graft_experts(model: TGAIV3Model, cfg_new: TGAIV3Config,
                  seed: int = 1234) -> TGAIV3Model:
    """★ 专家扩容嫁接：把小模型权重搬进大配置，新增专家的权重做扰动复制。

    保留原有语言能力的原理：
      · 原有专家权重**原样保留**（不重置）⇒ 已有能力不丢
      · 新增专家 = 从已有专家复制 + 小噪声 ⇒ 起点接近旧模型，继续训练只是"分化"
      · 路由器：旧输出的 logits 原样保留，新专家 logits 初始化为负偏置
        ⇒ 初期新专家几乎不被选中，模型行为≈旧模型，随后逐步分化

    限制：只支持**同 d_model** 的扩容（专家数/层数/FFN 维度可变）。
          若 d_model 变了，需要额外的维度映射（不在此函数内做，避免隐式降质）。
    """
    g = torch.Generator().manual_seed(seed)
    if cfg_new.d_model != model.config.d_model:
        raise ValueError('graft_experts 要求 d_model 相同（%d vs %d）；'
                         '改 d_model 请用 resize 流程'
                         % (model.config.d_model, cfg_new.d_model))
    if cfg_new.vocab_size != model.config.vocab_size:
        raise ValueError('graft_experts 要求 vocab_size 相同')

    new = TGAIV3Model(cfg_new)
    sd_old = model.state_dict()
    sd_new = new.state_dict()

    copied, grafted, fresh = 0, 0, 0
    for k, v in sd_new.items():
        if k in sd_old and sd_old[k].shape == v.shape:
            sd_new[k] = sd_old[k].clone()
            copied += 1
            continue
        # 专家扩容：旧 expert i 直接映射；新增专家从已拷贝的旧专家里挑一个复制+噪声
        if '.experts.' in k:
            try:
                parts = k.split('.')
                ei = int(parts[parts.index('experts') + 1])
            except Exception:
                fresh += 1
                continue
            old_ids = sorted({int(p.split('.')[p.split('.').index('experts') + 1])
                              for p in sd_old if '.experts.' in p})
            if not old_ids:
                fresh += 1
                continue
            src = old_ids[ei % len(old_ids)]
            k_src = k.replace('.experts.%d.' % ei, '.experts.%d.' % src)
            if k_src in sd_old and sd_old[k_src].shape == v.shape:
                w = sd_old[k_src].clone()
                w = w + torch.randn(w.shape, generator=g) * (0.02 * w.std().clamp_min(1e-6))
                sd_new[k] = w
                grafted += 1
                continue
        # 路由器扩容：旧列保留，新列给负偏置（初期不被选中）
        if k == 'router.weight' and 'router.weight' in sd_old:
            old_w = sd_old[k]
            n_old = old_w.shape[0]
            w = v.clone()
            w[:n_old] = old_w
            if v.shape[0] > n_old:
                w[n_old:] = old_w.mean(dim=0, keepdim=True).repeat(
                    v.shape[0] - n_old, 1) - 2.0
            sd_new[k] = w
            grafted += 1
            continue
        if k == 'expert_bias' and 'expert_bias' in sd_old:
            b = torch.zeros_like(v)
            nb = sd_old[k].numel()
            b[:nb] = sd_old[k]
            if v.numel() > nb:
                b[nb:] = -0.5
            sd_new[k] = b
            grafted += 1
            continue
        fresh += 1

    new.load_state_dict(sd_new, strict=True)
    print('[graft] 原样复制 %d 张量 / 嫁接 %d / 新建 %d' % (copied, grafted, fresh))
    return new


def resize_width(model: TGAIV3Model, cfg_new: TGAIV3Config,
                 seed: int = 1234, eps: float = 1e-4) -> TGAIV3Model:
    """★ d_model 扩展嫁接（0.86B d=1024 -> 1.5B d=1280 用）。

    为什么需要：graft_experts 明确只支持【同 d_model】扩容，并在 L1236 抛
    ValueError 提示「改 d_model 请用 resize 流程」—— 但该流程原先并不存在
    （2026-10-16 全盘搜索 0 命中）。本函数把它补上。

    设计原则（function-preserving）：
        扩展后**初始前向输出与旧模型几乎完全相同**（误差量级 eps=1e-4）。
        新增维度不填 0 而填极小噪声：填 0 输出严格不变但梯度恒为 0，
        新维度永远学不到东西；填 1e-4 噪声既保持输出近似不变，又打破对称。

    处理规则（按张量形状自动判定哪一维是 d_model）：
        · 1D 张量长度 == d_old           -> 新维度填 1.0（norm/scale 语义）
        · 2D 最后一维 == d_old           -> 扩输入维，新列填 eps（Linear.in）
        · 2D 第一维   == d_old           -> 扩输出维，新行填 eps（Linear.out）
        · Embedding (V, d_old)           -> 扩最后一维，新列填 eps
        · router.weight（第一维=专家数） -> 交由 graft_experts 处理，这里不动
        · 形状完全相同                   -> 原样复制

    限制：
        · vocab_size 必须相同（换词表是另一件事，需要重新映射 embedding）
        · 只支持**加宽**（d_new > d_old），缩窄不在本函数职责内
        · 专家数/层数的变化请配合 graft_experts / graft_layers 使用
    """
    if cfg_new.vocab_size != model.config.vocab_size:
        raise ValueError('resize_width 要求 vocab_size 相同（%d vs %d）；'
                         '换词表请走独立的 embedding 重映射流程'
                         % (model.config.vocab_size, cfg_new.vocab_size))
    d_old = model.config.d_model
    d_new = cfg_new.d_model
    if d_new < d_old:
        raise ValueError('resize_width 只用于加宽（%d -> %d）' % (d_old, d_new))
    if d_new == d_old:
        # 只变了别的字段 —— 退化成 graft_experts 的职责
        return graft_experts(model, cfg_new, seed=seed)

    g = torch.Generator().manual_seed(seed)
    new = TGAIV3Model(cfg_new)
    sd_old = model.state_dict()
    sd_new = new.state_dict()

    copied = widened = fresh = 0
    for k, v in sd_new.items():
        src = sd_old.get(k)
        if src is None:
            fresh += 1
            continue
        if src.shape == v.shape:
            sd_new[k] = src.clone()
            copied += 1
            continue

        # router 的第一维是专家数，与 d_model 无关 —— 留给 graft_experts
        if k.endswith('router.weight') or k.endswith('expert_bias'):
            fresh += 1
            continue

        if src.dim() == 1:
            # norm / scale：新维度填 1.0（保持数值尺度不变）
            if v.shape[0] != d_new or src.shape[0] != d_old:
                fresh += 1
                continue
            nw = torch.ones_like(v)
            nw[:d_old] = src
            nw[d_old:] = 1.0
            sd_new[k] = nw
            widened += 1
            continue

        if src.dim() == 2:
            # 最后一维是 d_old -> 扩输入维（新列）
            if src.shape[-1] == d_old and v.shape[-1] == d_new:
                nw = torch.empty_like(v)
                nw[:, :d_old] = src
                scale = float(src.std().clamp_min(1e-6)) * eps
                nw[:, d_old:] = torch.randn(
                    (v.shape[0], d_new - d_old), generator=g) * scale
                sd_new[k] = nw
                widened += 1
                continue
            # 第一维是 d_old -> 扩输出维（新行）
            if src.shape[0] == d_old and v.shape[0] == d_new:
                nw = torch.empty_like(v)
                nw[:d_old, :] = src
                scale = float(src.std().clamp_min(1e-6)) * eps
                nw[d_old:, :] = torch.randn(
                    (d_new - d_old, v.shape[1]), generator=g) * scale
                sd_new[k] = nw
                widened += 1
                continue

        fresh += 1

    new.load_state_dict(sd_new, strict=True)
    print('[resize_width] d_model %d -> %d : 原样 %d / 加宽 %d / 新建 %d'
          % (d_old, d_new, copied, widened, fresh))
    return new


def graft_layers(model: TGAIV3Model, cfg_new: TGAIV3Config,
                 seed: int = 1234) -> TGAIV3Model:
    """层数加深嫁接：新层从**相邻已有层**复制（而非随机初始化），保留语言能力。"""
    if cfg_new.n_layers <= model.config.n_layers:
        raise ValueError('graft_layers 只用于加深（%d → %d）'
                         % (model.config.n_layers, cfg_new.n_layers))
    # 先按专家扩容建好，再修层
    new = TGAIV3Model(cfg_new)
    sd_old = model.state_dict()
    sd_new = new.state_dict()
    n_old = model.config.n_layers
    g = torch.Generator().manual_seed(seed)
    for k, v in sd_new.items():
        if '.blocks.' in k:
            parts = k.split('.')
            li = int(parts[parts.index('blocks') + 1])
            src = min(li, n_old - 1)
            k_src = k.replace('.blocks.%d.' % li, '.blocks.%d.' % src)
            if k_src in sd_old and sd_old[k_src].shape == v.shape:
                w = sd_old[k_src].clone()
                if li >= n_old:      # 新层加噪声，打破对称
                    w = w + torch.randn(w.shape, generator=g) * (0.01 * w.std().clamp_min(1e-6))
                sd_new[k] = w
            continue
        if k in sd_old and sd_old[k].shape == v.shape:
            sd_new[k] = sd_old[k].clone()
    # 专家扩容部分（FFN 维度/专家数变化）交给 graft_experts 的逻辑
    tmp = TGAIV3Model(cfg_new)
    tmp.load_state_dict(sd_new, strict=True)
    if (cfg_new.n_experts != model.config.n_experts
            or cfg_new.d_ff != model.config.d_ff):
        tmp = graft_experts(model, cfg_new, seed)
        # 再把层嫁接覆盖回去
        sd2 = tmp.state_dict()
        for k, v in sd2.items():
            if '.blocks.' in k:
                parts = k.split('.')
                li = int(parts[parts.index('blocks') + 1])
                src = min(li, n_old - 1)
                k_src = k.replace('.blocks.%d.' % li, '.blocks.%d.' % src)
                if k_src in sd_old and sd_old[k_src].shape == v.shape:
                    w = sd_old[k_src].clone()
                    if li >= n_old:
                        w = w + torch.randn(w.shape, generator=g) * 0.01
                    sd2[k] = w
        tmp.load_state_dict(sd2, strict=True)
    return tmp


# ═══════════════════════════════════════════════════════════
# 便捷构造
# ═══════════════════════════════════════════════════════════
def build_meta(cfg: TGAIV3Config) -> TGAIV3Model:
    """★ 低内存构建：在 meta 设备上建模型（零内存占用）。

    用途：大模型（3B/7B）只需要**参数量统计**或**加载 checkpoint** 时，
    直接 `TGAIV3Model(cfg)` 会在 31.9 GB 内存的机器上被 OOM 杀掉
    （float32 初始化 + 逐层构造的峰值）。用 meta 设备可零成本拿到结构。
    之后用 `load_state_dict(..., assign=True)` 把权重灌进去。
    """
    import torch.nn as nn
    with torch.device('meta'):
        m = TGAIV3Model(cfg)
    return m


def create_model(preset_name: str = 'save_0_86b', **overrides) -> TGAIV3Model:
    cfg = preset(preset_name)
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError('配置里没有字段 %s' % k)
        setattr(cfg, k, v)
    return TGAIV3Model(cfg)


def human(n: int) -> str:
    if n >= 1e9:
        return '%.3fB' % (n / 1e9)
    if n >= 1e6:
        return '%.2fM' % (n / 1e6)
    return '%.1fK' % (n / 1e3)


if __name__ == '__main__':
    import sys
    names = sys.argv[1:] or ['w10m', 'w22m', 'save_0_5b', 'save_0_86b',
                             'save_1_5b', 'save_3b', 'v3_7b']
    print('=' * 104)
    print('%-12s %-10s %-10s %-8s %-5s %-8s %-6s %-8s %-8s %-8s %-6s' %
          ('预设', '总参数', '激活', 'd_model', '层', '专家', 'KV头', '词表',
           'd_ff', '上下文', 'AttnRes'))
    print('=' * 104)
    for n in names:
        try:
            # ★ 大模型用 meta 设备构建，避免 OOM（只统计参数量）
            cfg = preset(n)
            big = cfg.d_model * cfg.vocab_size > 100_000_000     # 词表嵌入 >400MB
            m = build_meta(cfg) if big else create_model(n)
            r = m.param_report()
            c = m.config
            tag = '稠密' if c.dense_ffn else '%d选%d' % (c.n_experts, c.n_activated)
            print('%-12s %-10s %-10s %-8d %-5d %-8s %-6d %-8d %-8d %-8d %-6s' %
                  (n, human(r['total']), human(r['active']), c.d_model, c.n_layers,
                   tag, c.kv_heads, c.vocab_size, c.d_ff, c.max_seq_len,
                   '开' if m.use_attn_res else '关'))
            if not c.dense_ffn:
                print('%-12s   每专家 %s · 注意力排布 %s (G=全局 c=CSA) · AttnRes=%s'
                      % ('', human(r['per_expert']),
                         ''.join('G' if c.is_global_layer(i) else 'c'
                                 for i in range(min(c.n_layers, 24))),
                         '开' if m.use_attn_res else '关'))
            del m
        except Exception as e:
            print('%-12s ❌ %s' % (n, str(e)[:70]))
    print('=' * 104)
