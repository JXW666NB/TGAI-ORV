# -*- coding: utf-8 -*-
"""训练检查点的三条契约：**v4 写出 / 分词器内置 / 优化器状态可往返**。

## 为什么要锁这三条

目标很具体：用户训出来的检查点要**自带优化器状态和分词器**，既能继续训、又能直接推理。
拆开就是三个可独立验证的断言：

1. `weights.tg` 的头里 `version == 4`（推理端 0.9.59 起认这个版本，且布局与 v3 相同）；
2. **内置分词表**：`read_tg()` 能从同一个文件解析出词表 —— 推理不需要旁边再放 sidecar；
3. **优化器状态可往返**：`save_opt_states` → `load_opt_states` 之后 `t/m/v` 逐位相同，
   否则续训等于从头再热身一遍。

第 1、2 条用真实夹具的 tok_blob，第 3 条用"只实现 state_dict/load_state_dict 的假优化器"，
所以**整个文件不需要 GPU**，能进 CI。
"""
import os

import numpy as np
import pytest

FIXTURE = r"D:\tgai_nlp\TGAI GO\tests\fixtures\mini_stream.TG"


def _have_fixture():
    return os.path.isfile(FIXTURE)


@pytest.mark.skipif(not _have_fixture(), reason="需要 TGAI GO 的 v3 测试夹具（提供真实 tok_blob）")
def test_ckpt_weights_are_v4_and_carry_the_tokenizer(tmp_path):
    from tgai_go.pack import TGWriter, DT_BF16
    from tgai_go.read_tg import read_tg
    from tgai_go.train.ckpt import read_tok_blob

    blob = read_tok_blob(FIXTURE)
    assert blob, "夹具里应能读出分词段"

    w = TGWriter(n_layers=1, d_model=16, n_heads=2, d_ff=32, vocab=32, max_seq=16,
                 n_experts=0, n_activated=0, rope_theta=10000, weight_dtype=DT_BF16)
    items = [("token_embedding.weight", np.zeros((32, 16), np.float32)),
             ("head.weight", np.zeros((32, 16), np.float32))]
    p = str(tmp_path / "weights.tg")
    w.write_stream(p, iter(items), version=4, tok_blob=blob)

    info = read_tg(p)
    assert info["header"]["version"] == 4, "检查点权重必须是 v4"
    assert info["header"]["n_layers"] == 1 and info["header"]["d_model"] == 16
    # ★ 分词器内置：同一个文件就能解析出词表，推理不必在旁边放 sidecar
    n_tok = len(info["tokens"])
    assert n_tok > 0, "weights.tg 应自带分词表（否则推理端还得找 sidecar）"


class _MockOpt(object):
    """只实现持久化契约的假优化器 —— 目的是测存取层，不测算法。"""

    def __init__(self, params, t=0, m=None, v=None):
        self.params = params
        self.t = t
        self.m = m if m is not None else [np.zeros_like(p) for p in params]
        self.v = v if v is not None else [np.zeros_like(p) for p in params]

    def state_dict(self):
        return {"t": self.t, "m": self.m, "v": self.v}

    def load_state_dict(self, sd):
        self.t = sd["t"]
        self.m = list(sd["m"])
        self.v = list(sd["v"])


def test_optimizer_state_roundtrip(tmp_path):
    """`t` / `m` / `v` 必须逐位往返 —— 差一点都等于续训没有动量。"""
    from tgai_go.train.opt_fp16mv import save_opt_states, load_opt_states

    rng = np.random.default_rng(0)
    params = [rng.normal(0, 0.1, (4, 3)).astype(np.float32),
              rng.normal(0, 0.1, (7,)).astype(np.float32)]
    m = [rng.normal(0, 0.01, p.shape).astype(np.float32) for p in params]
    v = [np.abs(rng.normal(0, 0.01, p.shape)).astype(np.float32) for p in params]
    src = _MockOpt(params, t=1234, m=m, v=v)

    d = str(tmp_path / "optim")
    os.makedirs(d)
    save_opt_states([src], d)

    dst = _MockOpt(params)                       # 全零起步，模拟"重新开进程"
    load_opt_states([dst], d)

    assert dst.t == 1234, "步数必须恢复（否则 LR schedule 会从头走）"
    for k, (a, b) in enumerate(zip(dst.m, m)):
        assert np.array_equal(a, b), "第 %d 个动量 m 不一致" % k
    for k, (a, b) in enumerate(zip(dst.v, v)):
        assert np.array_equal(a, b), "第 %d 个二阶矩 v 不一致" % k
