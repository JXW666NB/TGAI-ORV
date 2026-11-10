# -*- coding: utf-8 -*-
"""验证「分层即时更新」与「标准统一更新」**逐位一致**，并实测 3B 全参的显存。

## 为什么分层更新在数学上等价

反向是逐层的：层 i 的 `dL/dx_i` 一旦算完，**层 i-1 的反向不再需要层 i 的权重**。
所以在那个时刻用同一个 `t` 更新层 i，与"全部反向后统一更新"得到的参数完全一致。
好处是同时驻留的梯度只需一层的量（3B 上约 0.4 GiB，而不是 11.1 GiB）。

## 用法

    python layerwise_check.py equiv    # 等价性（小模型，快）
    python layerwise_check.py mem      # 3B 真实显存（慢，约 3~5 分钟）
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\tgai_orv\TGAI-ORV-main\python")

import importlib.util

try:
    _spec = importlib.util.spec_from_file_location("V27RUN", r"D:\tgai_orv\v27_train_run.py")
    _RUN = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_RUN)
    log = _RUN.log
except Exception:
    import datetime

    def log(msg):
        print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg), flush=True)

MODEL_TG = r"D:\tgai\models\V2.7\TGAI_NB_V2.7_v4.tg"
GIB = float(1 << 30)


# ═══════════════════════════════════════════════════════════
# A. 等价性：小模型，两轮同样的反向，比较"统一更新" vs "分层更新"
# ═══════════════════════════════════════════════════════════
def equiv():
    """同一初值、同一批数据：统一更新 vs 分层更新，参数必须**逐位相同**。"""
    import tgai_go.train as T
    from tgai_go.train.opt_fp16mv import Fp16MomAdamW

    T.set_default_device("cuda")
    rng = np.random.default_rng(0)
    V, S, D, NL = 64, 16, 32, 3
    model = T.preset.transformer(vocab=V, d_model=D, n_layers=NL, n_heads=4)
    params = list(model.parameters())
    sd0 = model.state_dict()                      # 初值快照
    ids = rng.integers(0, V, (S,), dtype=np.int64)
    tgt = rng.integers(0, V, (S,), dtype=np.int64)

    def run(per_layer):
        model.load_state_dict(sd0)                # 复位到同一初值
        opt = Fp16MomAdamW(params, lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95))
        loss = model(ids, tgt)
        loss.backward()
        if per_layer:
            # 只登记"每层的参数"；embed / final_norm / lm_head 由 auto_rest 自动进 -1 组
            groups = {i: list(model.blocks[i].parameters()) for i in range(NL)}
            opt.set_layer_groups(groups)          # 内部会补齐并校验覆盖完整
            for grp in opt.all_groups():          # 必须把 -1 组也走一遍
                opt.step_layers([grp], t=1, release=True)
        else:
            opt.step()
        T._cuda().sync()
        return model.state_dict()

    a = run(False)
    b = run(True)
    worst = 0.0
    for k in a:
        x = np.asarray(a[k], np.float32)
        y = np.asarray(b[k], np.float32)
        if x.size:
            worst = max(worst, float(np.abs(x - y).max()))
    log("统一更新 vs 分层更新：参数最大绝对差 %.3e ⇒ %s"
        % (worst, "逐位一致 ✔" if worst == 0.0 else "**不一致** ✘"))
    return 0 if worst == 0.0 else 1


# ═══════════════════════════════════════════════════════════
# B. 3B 真实显存：逐层反向 + 分层即时更新
# ═══════════════════════════════════════════════════════════
def mem():
    import torch
    import tgai_go.train as T
    from tgai_go.read_tg import close_tg, read_tg
    from tgai_go.train.opt_fp16mv import Fp16MomAdamW
    from tgai_go.train.v27_train import TGState, V27Trainer

    SEQ, NL = 512, 28
    info = read_tg(MODEL_TG); h = info["header"]; close_tg(info)
    D, NH = int(h["d_model"]), int(h["n_heads"])
    DH, THETA = D // NH, float(h["rope_theta"])
    cfg = dict(d_model=D, n_heads=NH, d_ff=int(h["d_ff"]), n_experts=int(h["n_experts"]),
               n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]), rope_theta=THETA)
    T.set_default_device("cuda")
    rt = T._cuda()
    total = 31.86
    low = [total]

    def sample(tag):
        f = rt.free_device_bytes() / GIB
        low[0] = min(low[0], f)
        log("  [%-14s] 空闲 %.2f GiB（峰值占用 %.2f GiB）" % (tag, f, total - low[0]))

    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=NL)
    tr.load_all(st)
    tr.attach_head(st)
    rt.empty_cache()
    sample("装载完")

    all_params = []
    groups = {}
    for i in range(NL):
        ps = tr.params_of(i)
        groups[i] = ps
        all_params.extend(ps)
    all_params.extend(list(tr.head_params()))
    log("参数张量 %d 个 / %.3f B" % (len(all_params), sum(int(p.numel()) for p in all_params) / 1e9))

    try:
        opt = Fp16MomAdamW(all_params, lr=1e-5, weight_decay=0.1)
        opt.set_layer_groups(groups)
        sample("优化器建好")

        rng = np.random.default_rng(0)
        ids = rng.integers(0, 30000, SEQ + 1, dtype=np.int64)
        j = np.arange(0, DH, 2, dtype=np.float32) / DH
        ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
        cos = T.tensor(np.cos(ang).astype(np.float32))
        sin = T.tensor(np.sin(ang).astype(np.float32))

        # ---------- 前向链（逐层落主机，不保留计算图）----------
        ys, xs = [], []
        x = T.tensor(rng.normal(0, 1, (SEQ, D)).astype(np.float32))
        for i in range(NL):
            ya = tr.fwd_attn(x, i, cos, sin)
            ys.append(np.asarray(ya.to_numpy(), np.float32))
            y = tr.moe_part(ya, i)
            xs.append(np.asarray(y.to_numpy(), np.float32))
            x = T.tensor(xs[-1])
            if i % 7 == 6:
                sample("前向 %d/%d" % (i + 1, NL))

        # ---------- 逐层反向 + 即时更新（梯度只留一层）----------
        gw = rng.normal(0, 0.01, (SEQ, D)).astype(np.float32)
        for i in range(NL - 1, -1, -1):
            y_t = T.tensor(xs[i], requires_grad=True)
            z = tr.moe_part(y_t, i)
            (z * T.tensor(gw)).sum().backward()
            xi = T.tensor(ys[i], requires_grad=True)
            ya = tr.fwd_attn(xi, i, cos, sin)
            (ya * T.tensor(gw)).sum().backward()
            opt.step_layers([i], t=1, release=True)      # ← 立刻更新并释放梯度
            if i % 7 == 0:
                sample("反向 %d" % (NL - i))
        rt.sync()
        sample("完成")
        log("结论：3B 全参 AdamW **跑通**，峰值占用 %.2f GiB（空闲还剩 %.2f GiB）"
            % (total - low[0], low[0]))
    except Exception as exc:
        log("结论：仍然失败：%s: %s" % (type(exc).__name__, str(exc)[:200]))
    finally:
        try:
            st.close(); rt.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "mem"
    (equiv if mode == "equiv" else mem)()
