# -*- coding: utf-8 -*-
"""① 梯度重算的**可调档位** + ③ 无 Tensor Core 的启动警告（2026-11-21）。

起因：P100 复测报告显示 ORV 在该卡上不占优（B=2 快 18% 但显存 +26%，B=4 打平而 +37%）。
排查发现两件事：

  1. `--grad-ckpt`（每层不建图、反向重跑）**早就实现了**，但没有任何用户文档提到它，
     所以复测的人不知道有降显存的开关 —— 等于没有。而且它只有全开/全关，没有中间档。
  2. P100 是 cc=6.0（Pascal，无 Tensor Core）。ORV 的加速来自 tensor-core GEMM，
     在这类卡上不成立，但程序启动时**一句话都不说**，用户只会觉得"引擎慢"。
"""
import os

import pytest

from tgai_go.train import v3_pretrain as V3


# ---------------------------------------------------------------- ① 命令行档位
def _parser():
    return V3.build_arg_parser()


def test_grad_ckpt_layers_flag_exists_and_defaults_to_all():
    a = _parser().parse_args(["--data", "x", "--grad-ckpt"])
    assert a.grad_ckpt is True
    assert a.grad_ckpt_layers == 0, "默认 0 = 全部层都重算（保持旧行为）"


def test_grad_ckpt_layers_accepts_partial_value():
    a = _parser().parse_args(["--data", "x", "--grad-ckpt", "--grad-ckpt-layers", "3"])
    assert a.grad_ckpt_layers == 3


def test_grad_ckpt_cfg_carries_layer_limit(monkeypatch):
    """`_expand_cfg` 必须把档位写进 cfg，否则层循环读不到（静默失效）。"""
    a = _parser().parse_args(["--data", "x", "--grad-ckpt", "--grad-ckpt-layers", "2"])
    cfg = V3._expand_cfg(a)
    assert cfg.get("grad_ckpt") is True
    assert cfg.get("grad_ckpt_layers") == 2


def test_grad_ckpt_without_layers_means_all(monkeypatch):
    a = _parser().parse_args(["--data", "x", "--grad-ckpt"])
    cfg = V3._expand_cfg(a)
    assert cfg.get("grad_ckpt") is True
    assert not cfg.get("grad_ckpt_layers"), "不开档位时不要写这个键（等价全部重算）"


def test_no_grad_ckpt_keeps_cfg_clean():
    a = _parser().parse_args(["--data", "x"])
    cfg = V3._expand_cfg(a)
    assert not cfg.get("grad_ckpt")
    assert not cfg.get("grad_ckpt_layers")


# ------------------------------------------------------ ③ 无 Tensor Core 警告
CAPS_OLD = {"cc": (6, 0), "fp16_arith": False, "bf16_arith": False,
            "bf16_store": True, "fp16_store": True}
CAPS_NEW = {"cc": (8, 9), "fp16_arith": True, "bf16_arith": True,
            "bf16_store": True, "fp16_store": True}


def test_warns_on_pascal(monkeypatch, capsys):
    monkeypatch.delenv("TG_NO_GPU_WARN", raising=False)
    V3.warn_no_tensor_core(caps=CAPS_OLD)
    out = capsys.readouterr().out
    assert "没有 Tensor Core" in out
    assert "cc=6.0" in out
    assert "修法" in out, "警告必须给出修法"
    assert "P100" in out, "要指名具体卡型，用户才知道是不是自己"


def test_silent_on_tensor_core(monkeypatch, capsys):
    monkeypatch.delenv("TG_NO_GPU_WARN", raising=False)
    V3.warn_no_tensor_core(caps=CAPS_NEW)
    assert capsys.readouterr().out == ""


def test_can_be_silenced(monkeypatch, capsys):
    monkeypatch.setenv("TG_NO_GPU_WARN", "1")
    V3.warn_no_tensor_core(caps=CAPS_OLD)
    assert capsys.readouterr().out == ""


def test_probe_failure_never_blocks_training(monkeypatch, capsys):
    """探测不到 GPU（无卡/驱动异常）时必须静默放行，不能因此起不来。"""

    class Boom(object):
        def device_caps(self):
            raise RuntimeError("no cuda")

    monkeypatch.delenv("TG_NO_GPU_WARN", raising=False)
    assert V3.warn_no_tensor_core(rt=Boom()) is None
    assert capsys.readouterr().out == ""


def test_message_mentions_checkpoint_knob(monkeypatch, capsys):
    """警告里要告诉用户"还能加 --grad-ckpt 降显存"，否则只是抱怨。"""
    monkeypatch.delenv("TG_NO_GPU_WARN", raising=False)
    V3.warn_no_tensor_core(caps=CAPS_OLD)
    out = capsys.readouterr().out
    assert "--grad-ckpt" in out and "--grad-ckpt-layers" in out

def test_bf16_act_flags_exist():
    """① 之前 bf16 激活存储只能靠环境变量，用户发现不了；现补成 CLI。"""
    from tgai_go.train import v3_pretrain as _V3
    a = _V3.build_arg_parser().parse_args(["--data", "x"])
    assert a.bf16_act is False, "默认必须关闭（全有或全无的优化，需实测确认）"
    assert a.bf16_act_min == 262144
    b = _V3.build_arg_parser().parse_args(["--data", "x", "--bf16-act",
                                           "--bf16-act-min", "1024"])
    assert b.bf16_act is True and b.bf16_act_min == 1024
