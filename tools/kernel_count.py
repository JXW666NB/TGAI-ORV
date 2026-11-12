# -*- coding: utf-8 -*-
"""统计一层的前向/反向**发射了多少次 kernel**，并按内核名分类（找出小 kernel 大户）。

思路：给 rt.launch 装计数器（只统计名字），跑一层前向 + 反向，输出 Top-N。

## 这是**作者本机**的脚本，不是给所有用户跑的

它需要两个仓库之外的东西（V2.7 训练入口与一个 .tg 权重）：

    V27_RUN  作者本机的 v27_train_run.py（**不在仓库里**）
    V27_TG   V2.7 的 .tg 权重路径

2026-11-12 修（外部报告 P2）：原来把这两个路径写死在 `D:\\tgai_orv\\...`，
换台机器直接崩。现在改成从环境变量或命令行读，缺了就**明确告诉你要什么**。

运行（需独占 GPU）：

    set V27_RUN=D:\\path\\v27_train_run.py
    set V27_TG=D:\\path\\model.tg
    python kernel_count.py
"""
import os
import sys
from collections import Counter

import numpy as np

# 仓库根：本文件在 <repo>/tools/ 下
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.join(_REPO, "python")
if os.path.isdir(os.path.join(_PKG, "tgai_go")) and _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import importlib.util                     # noqa: E402


def _need(name, what):
    val = os.environ.get(name, "").strip()
    if not val:
        raise SystemExit(
            "%s 未设置（需要%s）。\n"
            "修法：本脚本是**作者本机**工具，依赖仓库外的 V2.7 训练入口与权重。\n"
            "  set %s=<v27_train_run.py 的绝对路径>\n"
            "  set V27_TG=<V2.7 .tg 权重的绝对路径>\n"
            "只想做通用性能分析的话，用 tools/profile_block.py（自带合成权重，无外部依赖）。"
            % (name, what, name))
    if not os.path.isfile(val):
        raise SystemExit(
            "%s 指向的文件不存在: %s\n修法：改成正确路径，或用 tools/profile_block.py（无外部依赖）"
            % (name, val))
    return val


RUN_PATH = _need("V27_RUN", "作者的 v27_train_run.py，不在仓库内")
MODEL_TG = _need("V27_TG", "V2.7 的 .tg 权重")

_spec = importlib.util.spec_from_file_location("V27RUN", RUN_PATH)
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
log = RUN.log

SEQ = 128
NL = 28


def main():
    import tgai_go.train as T
    from tgai_go.train.v27_train import TGState, V27Trainer
    from tgai_go.read_tg import close_tg, read_tg

    info = read_tg(MODEL_TG)
    h = info["header"]
    close_tg(info)
    DM, NH = int(h["d_model"]), int(h["n_heads"])
    DH, THETA = DM // NH, float(h["rope_theta"])
    cfg = dict(d_model=DM, n_heads=NH, d_ff=int(h["d_ff"]), n_experts=int(h["n_experts"]),
               n_activated=int(h["n_activated"]), n_layers=int(h["n_layers"]), rope_theta=THETA)
    T.set_default_device("cuda")
    st = TGState(MODEL_TG)
    tr = V27Trainer(cfg, n_layers=1)
    tr.load_all(st)
    rt = T._cuda()

    counts = Counter()
    orig_launch = type(rt).launch

    def counted(self, name, *a, **kw):
        counts[name] += 1
        return orig_launch(self, name, *a, **kw)

    type(rt).launch = counted

    rng = np.random.default_rng(0)
    ids = rng.integers(0, 30000, SEQ, dtype=np.int64)
    j = np.arange(0, DH, 2, dtype=np.float32) / DH
    ang = np.repeat(np.arange(SEQ, dtype=np.float32)[:, None] * (THETA ** (-j))[None, :], 2, axis=1)
    cos = T.tensor(np.cos(ang).astype(np.float32))
    sin = T.tensor(np.sin(ang).astype(np.float32))
    x0 = np.ascontiguousarray(np.asarray(st["token_embedding.weight"][ids], np.float32))

    # 预热（NVRTC 编译不计入统计）
    y = tr.fwd(T.tensor(x0), 0, cos, sin)
    (y * T.tensor(np.ones((SEQ, DM), np.float32))).sum().backward()
    rt.sync()

    counts.clear()
    x = T.tensor(x0, requires_grad=True)
    y = tr.fwd(x, 0, cos, sin)
    rt.sync()
    fwd_counts = Counter(counts)
    fwd_total = sum(fwd_counts.values())

    counts.clear()
    g = np.ones((SEQ, DM), np.float32)
    (y * T.tensor(g)).sum().backward()
    rt.sync()
    bwd_counts = Counter(counts)
    bwd_total = sum(bwd_counts.values())

    log("一层前向发射 %d 次；反向发射 %d 次 ⇒ 一步（28 层）约 %d 次"
        % (fwd_total, bwd_total, (fwd_total + bwd_total) * NL))
    log("--- 前向 Top10 ---")
    for name, c in fwd_counts.most_common(10):
        log("   %-28s %d" % (name, c))
    log("--- 反向 Top10 ---")
    for name, c in bwd_counts.most_common(10):
        log("   %-28s %d" % (name, c))
    st.close()
    rt.empty_cache()


if __name__ == "__main__":
    main()
