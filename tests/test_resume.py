# -*- coding: utf-8 -*-
"""续训存档 roundtrip（纯 CPU）：断了再练，轨迹必须和没断一样。

为什么这条测试重要：训练大模型可能跑几小时，机器重启/抢占是常态。
如果续训只恢复参数、不恢复 AdamW 的动量（m/v/t），轨迹会突变 ——
用户看到的是"续训后 loss 先跳一下"，然后归因到"引擎不稳定"。

覆盖：
1. 存档里有 `__step` / `__opt_t` / `__opt_m_*` / `__opt_v_*`；
2. 恢复后**从正确的步号继续**（warmup 不重跑、取样计划不错位）；
3. "3 步 + 存档 + 续 3 步" 的最后 3 步 loss == "一口气 6 步" 的最后 3 步。
"""
import importlib.util
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")          # 与其它 torch 用例保持同一依赖口径

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    import sys
    sys.path.insert(0, os.path.join(REPO, "python"))
    path = os.path.join(REPO, "examples", "qwen25_sft.py")
    spec = importlib.util.spec_from_file_location("qwen25_sft_resume", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen25_sft_resume"] = mod
    spec.loader.exec_module(mod)
    return mod


CFG = dict(V=64, D=32, L=1, NH=2, KV=1, FF=64, THETA=1e6, EPS=1e-6, TIE=True)
LR = 1e-3
SEQ = 24
BATCH = 2


def _mk(q):
    import tgai_go.train as T
    T.set_default_device("cpu")
    m = q.Qwen25(dict(CFG), q.make_tiny_weights(CFG, seed=1))
    opt = T.optim.AdamW(m.params(), lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    return m, opt


def test_resume_roundtrip_matches_uninterrupted(tmp_path):
    q = _load()
    seqs = q.tiny_sequences(dict(CFG), n=4, seq_len=SEQ, seed=2)

    # --- A) 一口气 6 步 ---
    m_a, opt_a = _mk(q)
    la, _ = q.sft(m_a, seqs, 6, LR, SEQ, clip=0.0, batch=BATCH, opt=opt_a, log_every=100)

    # --- B) 3 步 -> 存档 -> 新建模型+优化器 -> 续 3 步 ---
    m_b, opt_b = _mk(q)
    lb1, _ = q.sft(m_b, seqs, 3, LR, SEQ, clip=0.0, batch=BATCH, opt=opt_b, log_every=100)
    ck = str(tmp_path / "resume.npz")
    q.save_params(m_b, ck, opt=opt_b, step=3)

    z = np.load(ck, allow_pickle=False)
    assert "__step" in z.files and int(z["__step"][0]) == 3
    assert "__opt_t" in z.files and int(z["__opt_t"][0]) == 3
    assert "__opt_m_0" in z.files and "__opt_v_0" in z.files

    m_c, opt_c = _mk(q)                       # 与 B 同 seed 初始化，随后被存档覆盖
    st = q.load_params(m_c, ck, opt_c)
    assert st["step"] == 3 and st["opt"] is True and st["params"] > 0
    lb2, _ = q.sft(m_c, seqs, 3, LR, SEQ, clip=0.0, batch=BATCH,
                   start_step=3, opt=opt_c, log_every=100)

    # --- 判定：续训的 3 步 == 连续跑的末 3 步 ---
    assert len(la) == 6 and len(lb1) == 3 and len(lb2) == 3
    got = np.asarray(lb2, np.float64)
    want = np.asarray(la[3:], np.float64)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-6), (got, want)


def test_old_checkpoint_without_opt_still_loads(tmp_path):
    """老存档（只有参数）必须还能加载 —— 不能让格式升级把历史权重读废。"""
    q = _load()
    m, opt = _mk(q)
    p = str(tmp_path / "old.npz")
    np.savez(p, **{k: np.asarray(v.to_numpy(), np.float32) for k, v in m.p.items()})
    m2, opt2 = _mk(q)
    st = q.load_params(m2, p, opt2)
    assert st["params"] > 0
    assert st["opt"] is False and st["step"] == 0
