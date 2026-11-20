# -*- coding: utf-8 -*-
"""`tgai_go.train.v3_pretrain` 的 CPU 侧测试：数据层 / 存档 / 续训护栏。

这里**不碰 GPU**（本机没有 CUDA 也能跑）。GPU 上的"续训后 loss 与不中断一致"
是集成验证（见模块 docstring 的实测段），不是单元测试能替代的；
但真正容易出错的那几块（数据位置、顺序复现、名字/形状校验、调度器状态）
全部在 CPU 上可测，所以先在这里钉死。
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

import tgai_go.train as T                                        # noqa: E402
from tgai_go.train import v3_pretrain as M                       # noqa: E402
from tgai_go.train import v3_train_cli as CLI                    # noqa: E402


# ---------------------------------------------------------------- fixtures
def _write_npz(path, n_rows, width, vocab=1000, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab, (n_rows, width)).astype(np.int32)
    lab = ids.copy()
    seg = np.zeros((n_rows, width), np.int16)
    lab[:, 0] = M.IGNORE_INDEX
    lab[:, width // 2] = M.IGNORE_INDEX
    seg[:, width // 2:] = 1
    np.savez(path, input_ids=ids, labels=lab, seg_id=seg)


@pytest.fixture()
def shard_dir(tmp_path):
    d = tmp_path / "shards"
    d.mkdir()
    for i in range(3):
        _write_npz(str(d / ("part-%05d.npz" % i)), 7, 16, seed=i)
    return str(d)


def _corpus(shard_dir, seq=8, seed=7, vocab=1000):
    return M.PackedCorpus(M.expand_data_paths([shard_dir]), seq=seq, seed=seed,
                          vocab=vocab, verbose=False)


# ---------------------------------------------------------------- 数据层
def test_expand_data_paths_accepts_dir_file_and_glob(shard_dir, tmp_path):
    paths = M.expand_data_paths([shard_dir])
    assert len(paths) == 3 and all(p.endswith(".npz") for p in paths)
    one = M.expand_data_paths([paths[0]])
    assert one == [paths[0]]
    globbed = M.expand_data_paths([os.path.join(shard_dir, "part-0000[12].npz")])
    assert len(globbed) == 2
    with pytest.raises(SystemExit):
        M.expand_data_paths([str(tmp_path / "nope")])


def test_probe_npz_reads_header_only(shard_dir):
    p = sorted(os.listdir(shard_dir))[0]
    info = M.probe_shard(os.path.join(shard_dir, p), seq=8)
    assert info.fmt == "npz" and info.n_rows == 7 and info.width == 16
    assert info.n_windows == 7 * 2 and info.has_labels


def test_probe_npy2d_and_stream(tmp_path):
    a = np.arange(4 * 8, dtype=np.int32).reshape(4, 8) % 100
    f1 = tmp_path / "rows.npy"
    np.save(str(f1), a)
    info = M.probe_shard(str(f1), seq=8)
    assert info.fmt == "npy2d" and info.n_rows == 4 and info.n_windows == 4

    flat = (np.arange(35, dtype=np.uint32) % 100)
    f2 = tmp_path / "flat.bin"
    flat.tofile(str(f2))
    info2 = M.probe_shard(str(f2), seq=8, bin_dtype="uint32")
    assert info2.fmt == "bin" and info2.n_tokens == 35 and info2.n_windows == 4


def test_bin_without_dtype_is_rejected(tmp_path):
    f = tmp_path / "x.bin"
    (np.arange(16, dtype=np.uint16)).tofile(str(f))
    with pytest.raises(SystemExit) as e:
        M.probe_shard(str(f), seq=8)
    assert "--bin-dtype" in str(e.value)


def test_window_shapes_labels_and_seg(shard_dir):
    c = _corpus(shard_dir)
    ids, lab, seg = c.take(2)
    assert ids.shape == (2, 8) and lab.shape == (2, 8) and seg.shape == (2, 8)
    assert ids.dtype == np.int64 and seg.dtype == np.int64
    # labels 与 ids 同值（除 -100），seg 只取 0/1
    assert set(np.unique(seg)).issubset({0, 1})
    keep = lab != M.IGNORE_INDEX
    assert np.array_equal(lab[keep], ids[keep])


def test_stats_and_one_epoch_windows(shard_dir):
    c = _corpus(shard_dir)
    st = c.stats()
    assert st["shards"] == 3 and st["windows"] == 3 * 7 * 2
    assert st["tokens_per_epoch"] == st["windows"] * 8


def test_order_is_reproducible_with_same_seed(shard_dir):
    c1 = _corpus(shard_dir)
    c2 = _corpus(shard_dir)
    o1 = [c1.take(1)[0][0, 0] for _ in range(12)]
    o2 = [c2.take(1)[0][0, 0] for _ in range(12)]
    assert o1 == o2, "同一 seed 两次顺序不一致 ⇒ 不可复现"
    c3 = _corpus(shard_dir, seed=8)
    o3 = [c3.take(1)[0][0, 0] for _ in range(12)]
    assert o1 != o3, "换 seed 顺序应当改变"


def test_resume_state_continues_identically(shard_dir):
    base = _corpus(shard_dir)
    ref = [base.take(1)[0][0, 0] for _ in range(12)]

    cut = _corpus(shard_dir)
    head = [cut.take(1)[0][0, 0] for _ in range(5)]
    st = json.loads(json.dumps(cut.state_dict()))       # 走一遍 JSON，模拟落盘

    cont = _corpus(shard_dir)
    cont.load_state_dict(st)
    tail = [cont.take(1)[0][0, 0] for _ in range(7)]
    assert head == ref[:5]
    assert tail == ref[5:], "从断点恢复的数据位置与不中断不一致"


def test_epoch_rollover_wraps_and_reports(shard_dir):
    c = _corpus(shard_dir)
    n = c.total_windows
    for _ in range(n):
        c.take(1)
    # 位置是**懒推进**的：取完最后一个窗口后 offset 已到末尾，但 epoch 还没跳。
    # 这个状态本身可以落盘并在续训时正确翻页（见下一行），所以不去强制改成"立即翻页"。
    assert c.consumed == n and c.epoch == 0 and c.offset == c.window_order(0, c.rank).size
    c.take(1)
    assert c.epoch == 1 and c.rank == 0 and c.offset == 1 and c.consumed == n + 1


def test_state_dict_rejects_seq_and_shard_mismatch(shard_dir):
    c = _corpus(shard_dir)
    st = c.state_dict()
    bad_seq = dict(st, seq=999)
    with pytest.raises(SystemExit) as e:
        c.load_state_dict(bad_seq)
    assert "--seq" in str(e.value)
    bad_n = dict(st, n_shards=99)
    with pytest.raises(SystemExit) as e2:
        c.load_state_dict(bad_n)
    assert "--data" in str(e2.value)


def test_vocab_guard_rejects_wrong_vocab(shard_dir):
    _write_npz(os.path.join(shard_dir, "part-00009.npz"), 2, 16, vocab=1000, seed=9)
    c = M.PackedCorpus(M.expand_data_paths([shard_dir]), seq=8, seed=7, vocab=64,
                       verbose=False)
    with pytest.raises(SystemExit) as e:
        c.check_vocab()
    assert "vocab" in str(e.value)


# ---------------------------------------------------------------- 参数遍历 / 存档
def _fake_P(n_layers=2):
    d, ff, E = 4, 6, 2
    rng = np.random.default_rng(0)

    def t(shape):
        return T.tensor(rng.normal(0, .1, shape).astype(np.float32), requires_grad=True)

    P = {"emb": t((10, d)), "ln_f": t((d,)), "head": t((10, d)), "layers": []}
    for _ in range(n_layers):
        P["layers"].append(dict(
            ln1=t((d,)), ln2=t((d,)), router=t((E, d)),
            wq=t((d, d)), wk=t((d, d)), wv=t((d, d)), wo=t((d, d)),
            W1=[t((ff, d)) for _ in range(E)],
            W2=[t((ff, d)) for _ in range(E)],
            W3=[t((d, ff)) for _ in range(E)],
            S1=t((ff, d)), S2=t((ff, d)), S3=t((d, ff)),
        ))
    return P


def test_param_slots_match_params_of_order_and_identity():
    P = _fake_P()
    got = [cont[key] for _n, cont, key in M.param_slots(P)]
    want = CLI.params_of(P)
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert a is b, "param_slots 的顺序必须与 params_of 完全一致（存档要靠它）"
    names = M.param_names(P)
    assert names[:3] == ["emb", "ln_f", "head"]
    assert "layers.0.W1.1" in names and "layers.1.wo" in names


def test_param_roundtrip_is_bit_exact_cpu(tmp_path):
    P = _fake_P()
    d = str(tmp_path / "params")
    n = M.save_params(d, P)
    assert n == len(M.param_names(P))
    before = [np.array(c[k].to_numpy(), np.float32) for _n, c, k in M.param_slots(P)]
    # 把权重改坏，再从盘上恢复
    for _n, c, k in M.param_slots(P):
        c[k].data[...] = 0.0
    M.load_params(d, P)
    after = [np.array(c[k].to_numpy(), np.float32) for _n, c, k in M.param_slots(P)]
    for b, a in zip(before, after):
        assert np.array_equal(b, a), "参数存取不是逐位一致"


def test_load_params_detects_name_mismatch(tmp_path):
    P = _fake_P()
    d = str(tmp_path / "params")
    M.save_params(d, P)
    # 改名一个文件 ⇒ 必须报"参数名不一致"，而不是静默错配
    victim = sorted(f for f in os.listdir(d) if f.endswith(".npy"))[5]
    os.rename(os.path.join(d, victim), os.path.join(d, victim.replace("layers.", "layersX.")))
    with pytest.raises(SystemExit):
        M.load_params(d, P)


def test_load_params_detects_count_mismatch(tmp_path):
    P = _fake_P(n_layers=2)
    d = str(tmp_path / "params")
    M.save_params(d, P)
    P2 = _fake_P(n_layers=3)
    with pytest.raises(SystemExit) as e:
        M.load_params(d, P2)
    assert "参数个数" in str(e.value)


def test_optimizer_roundtrip_cpu(tmp_path):
    P = _fake_P(n_layers=1)
    params = CLI.params_of(P)
    opt = T.optim.AdamW(params, lr=0.01)
    rng = np.random.default_rng(3)
    for i in range(len(opt.m)):
        opt.m[i] = rng.normal(0, 1, opt.m[i].shape).astype(np.float32)
        opt.v[i] = rng.normal(0, 1, opt.v[i].shape).astype(np.float32)
    opt.t = 17
    d = str(tmp_path / "opt")
    n = M.save_optimizer(d, opt)
    assert n == len(params)

    # 默认按 fp16 落盘（checkpoint 从 18 GB 降到 12 GB），所以往返有 fp16 量级误差
    opt2 = T.optim.AdamW(params, lr=0.01)
    M.load_optimizer(d, opt2)
    assert opt2.t == 17
    for a, b in zip(opt.m, opt2.m):
        assert np.allclose(a, b, rtol=2e-3, atol=2e-3), "m 往返误差超过 fp16 量级"
    for a, b in zip(opt.v, opt2.v):
        assert np.allclose(a, b, rtol=2e-3, atol=2e-3), "v 往返误差超过 fp16 量级"


def test_optimizer_roundtrip_fp32_exact(tmp_path):
    """half=False 时必须**逐位**相等（给需要逐位复现的场景留一条路）。"""
    P = _fake_P(n_layers=1)
    params = CLI.params_of(P)
    opt = T.optim.AdamW(params, lr=0.01)
    rng = np.random.default_rng(5)
    for i in range(len(opt.m)):
        opt.m[i] = rng.normal(0, 1, opt.m[i].shape).astype(np.float32)
        opt.v[i] = rng.normal(0, 1, opt.v[i].shape).astype(np.float32)
    opt.t = 3
    d = str(tmp_path / "opt32")
    M.save_optimizer(d, opt, half=False)
    opt2 = T.optim.AdamW(params, lr=0.01)
    M.load_optimizer(d, opt2)
    assert opt2.t == 3
    for a, b in zip(opt.m, opt2.m):
        assert np.array_equal(a, b)
    for a, b in zip(opt.v, opt2.v):
        assert np.array_equal(a, b)


def test_save_run_layout_is_ckpt_compatible(tmp_path, shard_dir):
    c = _corpus(shard_dir)
    P = _fake_P(n_layers=1)
    opt = T.optim.AdamW(CLI.params_of(P), lr=0.01)
    sched = T.optim.lr_scheduler.CosineLR(opt, T_max=10, warmup_steps=1)
    sched.step()
    root = str(tmp_path / "run")
    d = M.save_run(root, 3, c, P, opt, sched,
                   state_extra={"seed": 7, "model": {"d": 4}, "schedule": {"t_max": 9}})
    assert os.path.isfile(os.path.join(d, "meta.json"))
    assert os.path.isfile(os.path.join(d, "train_state.json"))
    # ckpt.list_checkpoints / rotate 必须能认出它（目录约定兼容）
    from tgai_go.train import ckpt
    cks = ckpt.list_checkpoints(root)
    assert cks and cks[-1][0] == 3 and cks[-1][1] == d
    st = json.load(open(os.path.join(d, "train_state.json"), encoding="utf-8"))
    assert st["step"] == 3 and st["kind"] == "ckpt"
    assert st["schedule_state"]["n"] == 1
    assert st["data"] == c.state_dict()


def test_resolve_resume_picks_latest(tmp_path, shard_dir):
    c = _corpus(shard_dir)
    P = _fake_P(n_layers=1)
    opt = T.optim.AdamW(CLI.params_of(P), lr=0.01)
    sched = T.optim.lr_scheduler.CosineLR(opt, T_max=10)
    root = str(tmp_path / "run")
    M.save_run(root, 2, c, P, opt, sched, state_extra={"seed": 7})
    M.save_run(root, 5, c, P, opt, sched, state_extra={"seed": 7})
    latest = M.resolve_resume(root)
    assert latest.endswith("step_000005")
    assert M.resolve_resume(latest) == latest
    with pytest.raises(SystemExit):
        M.resolve_resume(str(tmp_path / "empty"))


# ---------------------------------------------------------------- 护栏
def test_is_finite_guard():
    assert M._is_finite(0.0) and M._is_finite(-1e30)
    assert not M._is_finite(float("nan"))
    assert not M._is_finite(float("inf"))
    assert not M._is_finite(None)


def test_schedule_cfg_and_resume_check():
    class A:
        lr = 3e-4
        min_lr_ratio = 0.1
        warmup_steps = 5
        warmup_ratio = 0.5
    sc = M._sched_cfg(A(), 60)
    assert sc == {"base_lr": 3e-4, "eta_min": pytest.approx(3e-5),
                  "warmup_steps": 5, "t_max": 55, "horizon": 60}
    saved = {"schema": M.SCHEMA, "model": {"d": 4}, "seed": 7,
             "schedule": {"base_lr": 3e-4, "eta_min": 3e-5,
                          "warmup_steps": 5, "t_max": 55}}
    assert M.check_resume_state(saved, {"d": 4}, sc, 7) is True
    with pytest.raises(SystemExit) as e:
        M.check_resume_state(saved, {"d": 8}, sc, 7)
    assert "模型配置" in str(e.value)
    with pytest.raises(SystemExit) as e2:
        M.check_resume_state(saved, {"d": 4}, sc, 8)
    assert "seed" in str(e2.value)
    with pytest.raises(SystemExit) as e3:
        M.check_resume_state(saved, {"d": 4}, dict(sc, t_max=54), 7)
    assert "LR" in str(e3.value)
    # 换了 LR 计划：--reschedule 交给 main 判断，这里只保证校验函数本身严格
    with pytest.raises(SystemExit):
        M.check_resume_state({"schema": 999}, {"d": 4}, sc, 7)


def test_scheduler_state_roundtrip_matches():
    class Opt:
        lr = 3e-4
    s1 = T.optim.lr_scheduler.CosineLR(Opt(), T_max=20, eta_min=3e-5, warmup_steps=3)
    seq = [s1.step() for _ in range(15)]
    s2 = T.optim.lr_scheduler.CosineLR(Opt(), T_max=20, eta_min=3e-5, warmup_steps=3)
    for _ in range(6):
        s2.step()
    s2.load_state_dict({"n": 6, "lr": seq[5], "base_lr": 3e-4})
    tail = [s2.step() for _ in range(9)]
    assert np.allclose(tail, seq[6:]), "调度器状态恢复后 lr 序列必须一致"


# ---------------------------------------------------------------------------
# MoE 辅助项（2026-11-20：对齐 HOPE V3 的 aux_loss）
# ---------------------------------------------------------------------------

def test_aux_coefs_default_zero_and_validation(monkeypatch):
    """系数默认 0 ⇒ 训练损失与 1.0.7 逐位相同；非法值要显式报错，不能静默当 0。"""
    from tgai_go.train import v3_pretrain as VP

    monkeypatch.delenv("ORV_Z_LOSS", raising=False)
    monkeypatch.delenv("ORV_AUX_BAL", raising=False)
    assert VP._aux_coefs() == (0.0, 0.0)

    monkeypatch.setenv("ORV_Z_LOSS", "0.0001")
    monkeypatch.setenv("ORV_AUX_BAL", "0.0002")
    assert VP._aux_coefs() == (0.0001, 0.0002)

    monkeypatch.setenv("ORV_Z_LOSS", "banana")
    with pytest.raises(ValueError):
        VP._aux_coefs()

    monkeypatch.setenv("ORV_Z_LOSS", "-1")
    with pytest.raises(ValueError):
        VP._aux_coefs()


def test_router_lse_identity():
    """z-loss 依赖的恒等式：`mean_j(x_j - log_softmax_j) == logsumexp(x)`。

    `forward_loss` 用它来算 router 的 lse（不必新写 logsumexp 内核）。
    若这个恒等式在某次重构后不成立，z-loss 会**静默算错**，所以钉住。
    """
    import tgai_go.train as T

    rng = np.random.default_rng(0)
    for shape in ((8, 16), (32, 8), (5, 33)):
        x = rng.normal(size=shape).astype(np.float32)
        lg = T.tensor(x.copy())
        lp = T.functional.log_softmax(lg, axis=-1)
        d = (np.asarray(lg.data, np.float64) - np.asarray(lp.data, np.float64))
        lse = d.sum(axis=-1) / float(shape[1])
        ref = np.log(np.exp(x.astype(np.float64)).sum(axis=-1))
        assert np.allclose(lse, ref, atol=1e-3), (shape, np.abs(lse - ref).max())