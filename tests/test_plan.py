# -*- coding: utf-8 -*-
"""`plan.py` 的测试：解析、校验、算术、判定、拒绝。

这个模块是"轮椅"的脑子，所以每条判据都要钉住：
* 显存/通信是**纯算术** ⇒ 用实数对拍
* "值不值得"是**估算** ⇒ 只钉"会得出正确方向"，不钉小数
* **拒绝**必须真的拒绝（装不下 / 专家数超了 / rank 越界）
"""
import pytest

from tgai_go.train import plan as P


# ---------------- --place 解析 ----------------
def test_place_simple():
    assert P.parse_place("0:e=4,1:e=4", 2) == {0: {"e": 4}, 1: {"e": 4}}


def test_place_with_layers():
    got = P.parse_place("0:e=6;layers=0-13,1:e=2;layers=14-27", 2)
    assert got[0] == {"e": 6, "layers": (0, 13)}
    assert got[1] == {"e": 2, "layers": (14, 27)}


def test_place_single_layer_number():
    assert P.parse_place("0:layers=7", 2)[0]["layers"] == (7, 7)


def test_place_empty_is_noop():
    assert P.parse_place("", 2) == {}
    assert P.parse_place(None, 2) == {}


@pytest.mark.parametrize("spec,frag", [
    ("0:e=4,3:e=4", "超出范围"),          # rank 3 但 world=2
    ("0:e=4,0:e=2", "写了两次"),
    ("0:layers=13-0", "区间反了"),
    ("0:foo=1", "不认识的规则"),
    ("0:e=abc", "专家数看不懂"),
])
def test_place_rejects_bad_input(spec, frag):
    with pytest.raises(ValueError) as e:
        P.parse_place(spec, 2)
    assert frag in str(e.value)


def test_place_error_has_fix_hint():
    """报错必须带修法 —— 这是本仓库的既定风格。"""
    with pytest.raises(ValueError) as e:
        P.parse_place("0:foo=1", 2)
    assert "修法" in str(e.value)


# ---------------- 显存算术 ----------------
def test_memory_matches_hand_arithmetic():
    """848.6M / int8gpu：(4 + 4 + 3.06) B/参数 ⇒ 8.74 GiB（不含激活）。"""
    gib = P.estimate_memory(848_645_257, "int8gpu", 1)[0]
    assert gib == pytest.approx(11.06 * 848_645_257 / (1 << 30), rel=1e-9)
    assert 8.7 < gib < 8.8


def test_memory_sharding_halves_the_right_parts():
    n = 3_000_000_000
    full = P.estimate_memory(n, "int8gpu", 2)[0]
    z1 = P.estimate_memory(n, "int8gpu", 2, shard_opt=True)[0]
    z2 = P.estimate_memory(n, "int8gpu", 2, shard_opt=True, shard_grad=True)[0]
    z3 = P.estimate_memory(n, "int8gpu", 2, shard_opt=True, shard_grad=True,
                           shard_param=True)[0]
    assert full > z1 > z2 > z3
    # 每档省的字节数正好是被分片的那部分的一半
    assert (full - z1) * (1 << 30) == pytest.approx(3.06 * n / 2, rel=1e-6)
    assert (z1 - z2) * (1 << 30) == pytest.approx(4.0 * n / 2, rel=1e-6)
    assert (z2 - z3) * (1 << 30) == pytest.approx(4.0 * n / 2, rel=1e-6)


def test_memory_reproduces_the_measured_3b_boundary():
    """实测：3B 在 2×31.4 GiB 上 DP 装不下（33.2 GiB/卡），ZeRO-1 起才行。

    这条是规划器可信度的核心 —— 它必须重现实测结论，否则就是装饰。
    """
    n = 3_000_000_000
    cap = 31.36 * 0.95
    assert P.estimate_memory(n, "int8gpu", 2)[0] > cap          # DP 装不下
    assert P.estimate_memory(n, "int8gpu", 2, shard_opt=True)[0] < cap   # ZeRO-1 可以


def test_memory_unknown_optim_raises():
    with pytest.raises(ValueError) as e:
        P.estimate_memory(1000, "sgd_magic", 1)
    assert "修法" in str(e.value)


def test_memory_place_experts_over_total_raises():
    with pytest.raises(ValueError) as e:
        P.estimate_memory(1000, "int8gpu", 2, place={0: {"e": 6}, 1: {"e": 6}},
                          n_experts=8)
    assert "超过模型实际专家数" in str(e.value)


# ---------------- 通信算术 ----------------
def test_comm_closes_with_the_measured_allreduce():
    """标定检查：848.6M 的 DP 通信必须算回实测的 ~140.8 ms。"""
    c = P.estimate_comm(848_645_257, 2, "dp")
    assert c["ms"] == pytest.approx(140.8, abs=1.0)


def test_comm_world1_is_zero():
    assert P.estimate_comm(1e9, 1, "dp")["ms"] == 0.0


def test_comm_ordering():
    n = 848_645_257
    dp = P.estimate_comm(n, 2, "dp")["gb"]
    z1 = P.estimate_comm(n, 2, "zero1")["gb"]
    z3 = P.estimate_comm(n, 2, "zero3")["gb"]
    # ZeRO-1 比 DP 通信**更多**（reduce_scatter + all_gather），这点必须如实反映
    assert z1 > dp
    # ZeRO-3 与 ZeRO-1 在本模型下恰好相等：两者都是"搬 4 份参数字节"
    # （ZeRO-1 = 2×梯度 + 2×参数；ZeRO-3 = 前向/反向各 2×参数；梯度与参数同为 4 B/参数）
    # 这不是巧合也不是 bug，把口径差别记在测试里，免得以后误以为写错了。
    assert z3 == pytest.approx(z1, rel=1e-9)


def test_comm_unknown_mode_raises():
    with pytest.raises(ValueError):
        P.estimate_comm(1000, 2, "telepathy")


# ---------------- 判定与拒绝 ----------------
def _devs(n=2, gib=31.36):
    return [P.Device(i, "RTX 5090", gib) for i in range(n)]


def test_check_fit_refuses_when_over_capacity():
    p = P.Plan("dp", 2, [33.0, 33.0], {"gb": 1, "ms": 1}, _devs())
    p.check_fit()
    assert p.ok is False
    assert "装不下" not in p.reason          # 我们的措辞是"卡上可用只有"
    assert "修法" in p.reason


def test_judge_flags_net_loss_for_the_measured_seq512_case():
    """实测：seq512 单卡 0.1372 s、归约 145 ms ⇒ DP 吞吐比 0.87×（净亏）。

    规划器必须自己得出"净亏"这个结论。
    """
    comm = P.estimate_comm(848_645_257, 2, "dp")
    p = P.Plan("dp", 2, [8.74, 8.74], comm, _devs()).check_fit()
    p.judge(single_step_s=0.1372, overlap=False)
    assert any("净亏" in n for n in p.notes), p.notes


def test_judge_says_worth_it_for_long_sequence():
    """seq2048 单卡 0.2486 s：通信 140 ms 摊到 2 卡后仍能正收益。"""
    comm = P.estimate_comm(848_645_257, 2, "dp")
    p = P.Plan("dp", 2, [8.74, 8.74], comm, _devs()).check_fit()
    p.judge(single_step_s=0.2486, overlap=False)
    assert not any("净亏" in n for n in p.notes), p.notes


def test_judge_overlap_makes_it_better():
    comm = P.estimate_comm(848_645_257, 2, "dp")
    a = P.Plan("dp", 2, [8.74, 8.74], comm, _devs())
    b = P.Plan("dp", 2, [8.74, 8.74], comm, _devs())
    a.judge(0.1372, overlap=False)
    b.judge(0.1372, overlap=True)
    # 重叠后暴露时间归零 ⇒ 估算吞吐比必须更高
    assert "重叠" in b.notes[0]
    assert a.notes[0] != b.notes[0]


def test_format_contains_the_verdict():
    p = P.Plan("dp", 2, [8.74, 8.74], P.estimate_comm(848_645_257, 2, "dp"), _devs())
    p.check_fit().judge(0.1372)
    txt = p.format()
    assert "方案 dp" in txt and "通信" in txt and "净亏" in txt


# ---------------- choose / build ----------------
def test_choose_picks_something_that_fits_for_3b():
    p = P.choose(3_000_000_000, _devs(), optim="int8gpu", seq=512,
                 single_step_s=0.30)
    assert p.ok, p.reason
    assert p.mode in ("zero1", "zero2", "zero3"), p.mode


def test_choose_flags_zero3_without_p2p():
    p = P.choose(11_000_000_000, _devs(2, 31.36), optim="adawm" if False else "adamw",
                 single_step_s=1.0, p2p=False)
    # 11B 只有 ZeRO-3 装得下；必须带上"没有 P2P 会很慢"的警告
    assert p.mode in ("zero3",)
    assert any("p2p=False" in n for n in p.notes), p.notes


class _A(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_build_with_place_spec():
    a = _A(dist="", place="0:e=4,1:e=4", n_params=848_645_257, opt="int8gpu",
            seq=512, d=1024, E=8, zero=0)
    p = P.build(a, devices=_devs())
    assert p.mode == "ep"
    assert any("显式分配" in n for n in p.notes)


def test_build_place_requires_experts_on_every_rank():
    a = _A(dist="", place="0:e=4,1", n_params=1000, opt="int8gpu", seq=512,
            d=1024, E=8, zero=0)
    with pytest.raises(ValueError) as e:
        P.build(a, devices=_devs())
    assert "没写专家数" in str(e.value)


def test_build_zero_flags_map_to_modes():
    for z, mode in ((0, "dp"), (1, "zero1"), (2, "zero2"), (3, "zero3")):
        a = _A(dist="2", place="", n_params=848_645_257, opt="int8gpu", seq=512,
                d=1024, E=8, zero=z)
        assert P.build(a, devices=_devs()).mode == mode
