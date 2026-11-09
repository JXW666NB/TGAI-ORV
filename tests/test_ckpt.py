# -*- coding: utf-8 -*-
"""检查点模块（CPU 可测部分）：meta、轮转、名字表读取。"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.train.ckpt import (  # noqa: E402
    CKPT_META, checkpoint_dir, list_checkpoints, read_names_sidecar, rotate)


def _mk(root, step):
    d = checkpoint_dir(root, step)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, CKPT_META), "w", encoding="utf-8") as f:
        json.dump({"step": step}, f)
    return d


def test_rotate_keeps_latest(tmp_path):
    root = str(tmp_path)
    for s in (200, 400, 600):
        _mk(root, s)
    removed = rotate(root, keep=2)
    left = [s for s, _ in list_checkpoints(root)]
    assert left == [400, 600], left
    assert len(removed) == 1


def test_rotate_noop_when_under_keep(tmp_path):
    root = str(tmp_path)
    _mk(root, 200)
    assert rotate(root, keep=2) == []
    assert [s for s, _ in list_checkpoints(root)] == [200]


def test_list_checkpoints_sorted(tmp_path):
    root = str(tmp_path)
    for s in (600, 200, 400):
        _mk(root, s)
    assert [s for s, _ in list_checkpoints(root)] == [200, 400, 600]


def test_names_sidecar_missing_is_none(tmp_path):
    assert read_names_sidecar(os.path.join(str(tmp_path), "nope.tg")) is None