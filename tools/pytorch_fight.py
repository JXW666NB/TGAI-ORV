# -*- coding: utf-8 -*-
"""PyTorch 这一侧的对手：用**同一份权重**、**同一前向语义**跑 3.03B 全参，量速度与显存。

三轮（对应"打一架"的三个问题）：

  A. 门 槛 —— PyTorch 全参 fp32 + AdamW 能不能在 32 GB 上站起来？
     （权重 12.1 + 梯度 12.1 + AdamW 两个 fp32 状态 24.2 = 48 GB ⇒ 预期 OOM，
       但要用实测峰值证明，不能空口说）
  B. 速 度 —— 前向 / 反向各自多久？与 ORV 的实测（前向 0.53 s、逐层反向 1.16 s）对比。
  C. 折 中 —— 换 Adafactor（因子化状态，显存小）能不能跑完整步？多久？

运行（在 V100 上，服务必须已停）：
    C:\\Python311\\python.exe -X utf8 D:\\tgai_orv\\pytorch_fight.py
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\tgai_orv")
import v27_ref  # noqa: E402

# v27_ref.rope_tables 默认在 CPU 建表（它本来是给单机对齐用的）；
# 这里只改「默认 device」，语义一个字不动。
_rope_orig = v27_ref.rope_tables
v27_ref.rope_tables = lambda seq, dh, theta=10000.0, device="cuda": _rope_orig(seq, dh, theta, device)

PT = r"D:\TGAI\models\V2.7\tgai_v27_v5_merged.pt"
SEQ = 512
CFG = dict(d_model=1024, n_heads=16, head_dim=64, n_experts=8, n_activated=2,
           rope_theta=10000.0)


def vram():
    return torch.cuda.memory_allocated() / 2**30, torch.cuda.max_memory_allocated() / 2**30


def build_params(sd, nl):
    """把 state_dict 装成 v27_ref 需要的 P 结构，全部 cuda fp32 + requires_grad。"""
    def P_(name):
        return torch.nn.Parameter(sd[name].float().to("cuda"), requires_grad=True)

    P = {"embed": P_("token_embedding.weight"), "final_norm": P_("final_norm.weight"),
         "blocks": []}
    for i in range(nl):
        p = "blocks.%d." % i
        b = {"ln1": P_(p + "ln1.weight"), "ln2": P_(p + "ln2.weight"),
             "q": P_(p + "attn.q_proj.weight"), "k": P_(p + "attn.k_proj.weight"),
             "v": P_(p + "attn.v_proj.weight"), "out": P_(p + "attn.out_proj.weight")}
        moe = {"router": P_(p + "moe.router.weight"), "experts": []}
        if (p + "moe._qb_bias") in sd:
            moe["qb_bias"] = P_(p + "moe._qb_bias")
        for e in range(CFG["n_experts"]):
            q = p + "moe.experts.%d." % e
            moe["experts"].append({"gate": P_(q + "w1.weight"),
                                   "up": P_(q + "w2.weight"),
                                   "down": P_(q + "w3.weight")})
        b["moe"] = moe
        P["blocks"].append(b)
    return P


def flatten(P):
    out = [P["embed"], P["final_norm"]]
    for b in P["blocks"]:
        out += [b["ln1"], b["ln2"], b["q"], b["k"], b["v"], b["out"],
                b["moe"]["router"]]
        if "qb_bias" in b["moe"]:
            out.append(b["moe"]["qb_bias"])
        for e in b["moe"]["experts"]:
            out += [e["gate"], e["up"], e["down"]]
    return out


def main():
    torch.backends.cuda.matmul.allow_tf32 = True     # V100 上 fp32 matmul 用 TF32（与 ORV 一致）
    torch.manual_seed(0)
    print("torch %s cuda %s %s" % (torch.__version__, torch.version.cuda,
                                   torch.cuda.get_device_name(0)))
    ck = torch.load(PT, map_location="cpu", mmap=True, weights_only=False)
    sd, mc = ck["model_state_dict"], ck["model_config"]
    nl = int(mc["n_layers"])
    print("模型 %d 层，目标 seq=%d" % (nl, SEQ))

    t0 = time.time()
    P = build_params(sd, nl)
    params = flatten(P)
    n_par = sum(p.numel() for p in params)
    a, mx = vram()
    print("装载 %.1fs：%.3f B 参数，显存 %.2f GB（峰值 %.2f）"
          % (time.time() - t0, n_par / 1e9, a, mx))

    g = torch.Generator(device="cpu").manual_seed(7)
    V = int(P["embed"].shape[0])
    ids = torch.randint(0, V, (SEQ,), generator=g).to("cuda")
    tgt = torch.randint(0, V, (SEQ,), generator=g).to("cuda")

    # ---- 轮 B：前向 / 反向 分别计时 ----
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    logits = v27_ref.forward(ids, P, CFG)
    t_fwd = time.time() - t0
    loss = F.cross_entropy(logits, tgt)
    a, mx = vram()
    print("前向 %.3fs  loss=%.4f  显存 %.2f GB（峰值 %.2f）" % (t_fwd, float(loss), a, mx))

    t0 = time.time()
    loss.backward()
    t_bwd = time.time() - t0
    a, mx = vram()
    print("反向 %.3fs  合计 %.3fs  显存 %.2f GB（峰值 %.2f）"
          % (t_bwd, t_fwd + t_bwd, a, mx))
    print("→ 对比 ORV 实测：前向 0.53s / 逐层反向 1.16s / 合计 1.69s（T=512）")

    # ---- 轮 A：AdamW 能不能站起来 ----
    print("\n=== 轮 A：PyTorch 全参 AdamW ===")
    try:
        opt = torch.optim.AdamW(params, lr=1e-6, betas=(0.9, 0.95), eps=1e-8)
        a, mx = vram()
        print("  优化器创建后 显存 %.2f GB（峰值 %.2f）" % (a, mx))
        t0 = time.time()
        opt.step()
        opt.zero_grad(set_to_none=True)
        a, mx = vram()
        print("  ✅ 站起来：step %.3fs，显存 %.2f GB（峰值 %.2f）"
              % (time.time() - t0, a, mx))
        del opt
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        a, mx = vram()
        print("  ❌ OOM：%s" % str(e).split("\n")[0][:120])
        print("  挂掉时显存 %.2f GB（峰值 %.2f）" % (a, mx))
    torch.cuda.empty_cache()

    # ---- 轮 C：Adafactor 完整步 ----
    print("\n=== 轮 C：PyTorch 全参 + Adafactor（因子化状态）===")
    for p in params:
        p.grad = None
    torch.cuda.reset_peak_memory_stats()
    try:
        # 不同 torch 版本 Adafactor 签名不同：先试完整参数，再退最小参数，
        # 最后退「无状态 SGD」并把差异写清楚（绝不把不同配置冒充同一配置）。
        opt = None
        opt_name = ""
        try:
            opt = torch.optim.Adafactor(params, lr=1e-6, scale_parameter=False,
                                        relative_step=False, warmup_init=False)
            opt_name = "Adafactor(完整参数)"
        except TypeError:
            try:
                opt = torch.optim.Adafactor(params, lr=1e-6)
                opt_name = "Adafactor(lr only)"
            except (AttributeError, TypeError):
                opt = torch.optim.SGD(params, lr=1e-6)
                opt_name = "SGD(无动量·无状态)"
        print("  优化器：%s" % opt_name)
        t0 = time.time()
        logits = v27_ref.forward(ids, P, CFG)
        loss = F.cross_entropy(logits, tgt)
        loss.backward()
        t_fb = time.time() - t0
        t0 = time.time()
        opt.step()
        opt.zero_grad(set_to_none=True)
        t_st = time.time() - t0
        a, mx = vram()
        print("  ✅ 完整步 %.3fs（前向+反向 %.3fs，优化器 %.3fs）loss=%.4f 峰值 %.2f GB"
              % (t_fb + t_st, t_fb, t_st, float(loss), mx))
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        a, mx = vram()
        print("  ❌ OOM：%s（峰值 %.2f GB）" % (str(e).split("\n")[0][:120], mx))
    print("\n完成")


if __name__ == "__main__":
    main()
