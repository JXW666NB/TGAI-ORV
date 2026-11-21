# -*- coding: utf-8 -*-
"""生成 ZeRO 验收用的 ``_zero_verify.py``：把 snippet 打进 **HEAD 版** v3_pretrain。

为什么生成副本而不是改 `v3_pretrain.py`：
  * 任务书要求不直接改 `v3_pretrain.py`（要把接线作为 snippet 交给 lead 合并）；
  * 工作区里的 `v3_pretrain.py` 正被另一个任务（DP 反向重叠）改，拿它当基底会漂。
所以基线取 `git show HEAD:python/tgai_go/train/v3_pretrain.py`（1.0.15 发布版，
已含 `--dp` 接线），再用**带断言的字符串替换**把 snippet 打进去。

用法::

    python lab/rent/zero_make_verify.py            # 生成 lab/rent/_zero_verify.py
    python lab/rent/zero_make_verify.py --stdout   # 打到 stdout（自检用）
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))          # ...\\_tgai_orv
REL = "python/tgai_go/train/v3_pretrain.py"
OUT = os.path.join(HERE, "_zero_verify.py")

# ═════════════════════════════════════════════════════════════════════════
# snippet：每个 (旧串, 新串) 都必须**恰好命中 1 次**，否则当场失败
# （宁可生成失败，也不要悄悄生成一个接线不对的副本）
# ═════════════════════════════════════════════════════════════════════════
EDITS = []

# ---- 1) 新开关 --zero ----
EDITS.append((
    '''    g.add_argument("--dp", type=int, default=0, metavar="N",
                   help="数据并行 world 大小（0 = 读 TGAI_DP_WORLD，默认 1 = 单卡）。"
                        "rank 仍由 TGAI_DP_RANK 指定，且每个进程必须只看见一张卡"
                        "（CUDA_VISIBLE_DEVICES=0 / 1）。归约后端由 TG_DP_BACKEND="
                        "host|nccl|auto 选（默认 host）。world=1 时本开关无任何影响")
''',
    '''    g.add_argument("--dp", type=int, default=0, metavar="N",
                   help="数据并行 world 大小（0 = 读 TGAI_DP_WORLD，默认 1 = 单卡）。"
                        "rank 仍由 TGAI_DP_RANK 指定，且每个进程必须只看见一张卡"
                        "（CUDA_VISIBLE_DEVICES=0 / 1）。归约后端由 TG_DP_BACKEND="
                        "host|nccl|auto 选（默认 host）。world=1 时本开关无任何影响")
    # ---- ZeRO（优化器状态/梯度分片；省显存，不省通信）----
    g.add_argument("--zero", type=int, default=0, choices=(0, 1, 2), metavar="0|1|2",
                   help="0=不分片（数据并行现状）；1=优化器状态按 rank 分片；"
                        "2=状态+梯度都分片（梯度归约用 reduce_scatter）。"
                        "需要 world>=2（--dp N 或 TGAI_DP_WORLD），"
                        "只支持 --opt adamw/fp16mom/int8gpu；"
                        "ZeRO-1 的通信量比 DP 多 50%%（不是更省），ZeRO-2 与 DP 持平 —— "
                        "它省的是显存，不是通信")
''',
))

# ---- 2) 优化器：用分片后的参数构造 + 建 ZeroTrainer ----
EDITS.append((
    '''    # ---- 优化器 / 调度器 ----
    if a.opt == "adamw":''',
    '''    # ---- 优化器 / 调度器 ----
    # ---- ZeRO（可选）：--zero 1/2。状态只分给 world 个 rank 各 1/w 份 ----
    _zt = None
    _zparams = params
    if int(getattr(a, "zero", 0) or 0) > 0:
        from tgai_go.train import zero as _Z
        _zt = _Z.ZeroTrainer(params, zero=int(a.zero), opt_name=a.opt)
        _zparams = _zt.shard_params          # ★ 优化器只对「自己那份」建状态
        print("[zero] " + _zt.describe(), flush=True)
    if a.opt == "adamw":''',
))

for _old, _new in [
    ("opt = T.optim.AdamW(params, lr=float(a.lr), betas=(float(a.beta1), float(a.beta2)),",
     "opt = T.optim.AdamW(_zparams, lr=float(a.lr), betas=(float(a.beta1), float(a.beta2)),"),
    ("opt = T.optim.Muon(params, lr=float(a.muon_lr), momentum=float(a.muon_momentum),",
     "opt = T.optim.Muon(_zparams, lr=float(a.muon_lr), momentum=float(a.muon_momentum),"),
    ("opt = Int8AdamWGpu(params, lr=float(a.lr),",
     "opt = Int8AdamWGpu(_zparams, lr=float(a.lr),"),
    ("opt = Fp16MomAdamW(params, lr=float(a.lr), betas=(float(a.beta1), float(a.beta2)),",
     "opt = Fp16MomAdamW(_zparams, lr=float(a.lr), betas=(float(a.beta1), float(a.beta2)),"),
    ("opt = T.optim.Adafactor(params, lr=float(a.lr), weight_decay=float(a.wd))",
     "opt = T.optim.Adafactor(_zparams, lr=float(a.lr), weight_decay=float(a.wd))"),
    ("opt = Int8AdamW(params, lr=float(a.lr), betas=(float(a.beta1), float(a.beta2)),",
     "opt = Int8AdamW(_zparams, lr=float(a.lr), betas=(float(a.beta1), float(a.beta2)),"),
]:
    EDITS.append((_old, _new))

# ---- 3) 存档/续训与 ZeRO 互斥（状态只存了 1/w，直接续训会静默错）----
EDITS.append((
    '''    _defer_clip = (a.opt == "int8gpu")''',
    '''    _defer_clip = (a.opt == "int8gpu")
    if _zt is not None and (int(a.ckpt_every) > 0 or a.resume):
        raise SystemExit(
            "--zero 与 --ckpt-every/--resume 暂不兼容：checkpoint 只实现了不分片状态的"
            "落盘，分片后若按原来的格式存/读会**静默**错位"
            "。修法：验证阶段用 --zero 但不存档（去掉 --ckpt-every/--resume）；"
            "要存分片 checkpoint 需另行实现（见 docs/ZERO_PLAN.md 的落地顺序）")''',
))

# ---- 4) 优化器外壳（zero_grad 清全部参数 + lr 透传）----
EDITS.append((
    '''    print("[sched] CosineLR warmup=%d t_max=%d eta_min=%.3e horizon=%d"
          % (sc["warmup_steps"], sc["t_max"], sc["eta_min"], sc["horizon"]), flush=True)
''',
    '''    print("[sched] CosineLR warmup=%d t_max=%d eta_min=%.3e horizon=%d"
          % (sc["warmup_steps"], sc["t_max"], sc["eta_min"], sc["horizon"]), flush=True)
    if _zt is not None:
        opt = _zt.wrap(opt)                  # 外壳：zero_grad 清**全部**参数 + lr 透传
''',
))

# ---- 5) DataParallel 建好 → attach ----
EDITS.append((
    '''        if _dp.rank == 0:
            print("[dp] world=%d rank=%d job=%s dir=%s backend=%s"
                  % (_dp.world, _dp.rank, _dp.job, _dp.dir, _dp.backend), flush=True)
''',
    '''        if _dp.rank == 0:
            print("[dp] world=%d rank=%d job=%s dir=%s backend=%s"
                  % (_dp.world, _dp.rank, _dp.job, _dp.dir, _dp.backend), flush=True)
        if _zt is not None:
            _zt.attach(_dp)                  # world/rank 与 TGAI_DP_* 不一致会当场报
''',
))

# ---- 6) 归约：ZeRO-1 = all_reduce（复用 DP）/ ZeRO-2 = reduce_scatter ----
EDITS.append((
    '''            _comm_s = None
            if _dp is not None:
                _t_r = _now()
                _nred = sum(1 for q in params if getattr(q, "grad", None) is not None)
                _bytes = sum(int(q.grad._gpu.n) * 4 for q in params
                             if getattr(q, "grad", None) is not None)
                _dp.all_reduce_grads(params)
                _comm_s = _now() - _t_r
                if os.environ.get("TG_DP_TIMING"):
                    print("[dp] reduce %.4f s  grads=%d  bytes=%.2f GiB  => %.1f GB/s"
                          "[%s]"
                          % (_comm_s, _nred, _bytes / float(1 << 30),
                             (_bytes / _comm_s) / 1e9 if _comm_s > 0 else 0.0,
                             _dp.backend), flush=True)
''',
    '''            _comm_s = None
            if _zt is not None:
                # ★ 必须在裁剪**之前**归约（各 rank 的 gnorm/裁剪阈值才会一致）
                _comm_s = _zt.reduce_grads()
            elif _dp is not None:
                _t_r = _now()
                _nred = sum(1 for q in params if getattr(q, "grad", None) is not None)
                _bytes = sum(int(q.grad._gpu.n) * 4 for q in params
                             if getattr(q, "grad", None) is not None)
                _dp.all_reduce_grads(params)
                _comm_s = _now() - _t_r
                if os.environ.get("TG_DP_TIMING"):
                    print("[dp] reduce %.4f s  grads=%d  bytes=%.2f GiB  => %.1f GB/s"
                          "[%s]"
                          % (_comm_s, _nred, _bytes / float(1 << 30),
                             (_bytes / _comm_s) / 1e9 if _comm_s > 0 else 0.0,
                             _dp.backend), flush=True)
''',
))

# ---- 7) gnorm / 裁剪：ZeRO-2 用分布式平方和复现同一数值 ----
EDITS.append((
    '''            gscale = 1.0
            if float(a.grad_clip) > 0:
                if _defer_clip:
                    # 不物理缩放；scale 交给 opt.step 在内核里乘（省一趟 读+写）。
                    gnorm, gscale = T.clip_grad_norm_(
                        params, float(a.grad_clip), defer_scale=True)
                    gnorm = float(gnorm)
                    gscale = float(gscale)
                else:
                    gnorm = float(T.clip_grad_norm_(params, float(a.grad_clip)))
            else:
                gnorm = 0.0
''',
    '''            gscale = 1.0
            if float(a.grad_clip) > 0:
                if _zt is not None:
                    # ZeRO-1：梯度是完整的（与 DP 同一条路径）；
                    # ZeRO-2：梯度被切开 ⇒ 用「分块槽位 + 固定顺序树归约」复现同一数值
                    gnorm, gscale = _zt.grad_norm_scale(
                        float(a.grad_clip), defer=_defer_clip)
                    gnorm = float(gnorm)
                    gscale = float(gscale) if _defer_clip else 1.0
                elif _defer_clip:
                    # 不物理缩放；scale 交给 opt.step 在内核里乘（省一趟 读+写）。
                    gnorm, gscale = T.clip_grad_norm_(
                        params, float(a.grad_clip), defer_scale=True)
                    gnorm = float(gnorm)
                    gscale = float(gscale)
                else:
                    gnorm = float(T.clip_grad_norm_(params, float(a.grad_clip)))
            else:
                gnorm = 0.0
''',
))

# ---- 8) 更新参数后 all_gather（每卡恢复成完整且逐位相同的参数）----
EDITS.append((
    '''            if _defer_clip:
                opt.step(grad_scale=gscale)
            else:
                opt.step()
''',
    '''            if _defer_clip:
                opt.step(grad_scale=gscale)
            else:
                opt.step()
            if _zt is not None:
                _zt.after_step()             # all_gather 参数（★ 在 AMP 失效之前）
''',
))

# ---- 9) RESULT 里报 ZeRO 的统计（通信量/分片分布）----
EDITS.append((
    '''        "dp": (_dp.stats() if _dp is not None else None),''',
    '''        "dp": (_dp.stats() if _dp is not None else None),
        "zero": (_zt.describe() if _zt is not None else None),
        "zero_stats": (_zt.stats.as_dict() if _zt is not None else None),
        "zero_comm_model": (_Z.comm_model(_zt.plan.total_elems, _zt.plan.world,
                                          _zt.zero) if _zt is not None else None),''',
))

# ---- 10) 通信器关闭时顺带清掉 ZeRO 的共享内存槽 ----
EDITS.append((
    '''    if _dp is not None:
        _dp.close()
''',
    '''    if _zt is not None:
        _zt.close()
    if _dp is not None:
        _dp.close()
''',
))

# ---- 11) 副本在包外：把 main 里唯一的相对 import 改成绝对 import ----
EDITS.append((
    '''        from . import dist as _dist_mod''',
    '''        from tgai_go.train import dist as _dist_mod''',
))


def build():
    src = subprocess.check_output(
        ["git", "show", "HEAD:" + REL], cwd=REPO).decode("utf-8")
    out = src
    for i, (old, new) in enumerate(EDITS):
        n = out.count(old)
        if n != 1:
            raise SystemExit(
                "snippet 第 %d 处锚点命中 %d 次（要求恰好 1 次）—— HEAD 基线与预期不一致。\n"
                "锚点首行：%r" % (i + 1, n, old.splitlines()[0] if old else ""))
        out = out.replace(old, new, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    a = ap.parse_args()
    text = build()
    if a.stdout:
        sys.stdout.write(text)
        return 0
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("[ok] wrote %s (%d bytes, %d edits)" % (OUT, len(text.encode("utf-8")),
                                                 len(EDITS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
