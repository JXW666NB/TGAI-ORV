# -*- coding: utf-8 -*-
"""names 边车回退链（v4 无自己边车时用同族 v3 的，但绝不乱配）。"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train.ckpt import names_sidecar_path, read_names_sidecar  # noqa: E402


def _write_sidecar(tg_path, names):
    with open(tg_path + ".names.json", "w", encoding="utf-8") as f:
        json.dump({"tensors": [{"name": n} for n in names]}, f)


def test_exact_sidecar_wins(tmp_path):
    tg = str(tmp_path / "M_v4.tg")
    open(tg, "wb").close()
    _write_sidecar(tg, ["a", "b"])
    assert names_sidecar_path(tg) == tg + ".names.json"
    assert read_names_sidecar(tg) == ["a", "b"]


def test_fallback_to_same_family_v3(tmp_path):
    v3 = str(tmp_path / "TGAI_NB_V2.7.tg")
    v4 = str(tmp_path / "TGAI_NB_V2.7_v4.tg")
    open(v3, "wb").close()
    open(v4, "wb").close()
    _write_sidecar(v3, ["x", "y", "z"])
    assert names_sidecar_path(v4) == v3 + ".names.json"
    assert read_names_sidecar(v4) == ["x", "y", "z"]


def test_ambiguous_returns_none(tmp_path):
    v4 = str(tmp_path / "Model_v4.tg")
    open(v4, "wb").close()
    for name in ("ModelA.tg", "ModelB.tg"):
        p = str(tmp_path / name)
        open(p, "wb").close()
        _write_sidecar(p, ["n"])
    # 前缀相同但两个候选 ⇒ 不猜
    os.rename(str(tmp_path / "ModelA.tg.names.json"), str(tmp_path / "ModelX1.tg.names.json"))
    os.rename(str(tmp_path / "ModelB.tg.names.json"), str(tmp_path / "ModelX2.tg.names.json"))
    assert names_sidecar_path(v4) is None
    assert read_names_sidecar(v4) is None