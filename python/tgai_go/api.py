# -*- coding: utf-8 -*-
"""十行代码训练的入口：`from tgai_go import api; api.train(...)`。

设计取舍（如实说明）
--------------------
v1 是**编排层**：它校验参数、拼出命令、启动仓库里**已验证**的训练脚本
（`examples/qwen25_sft.py`），再把结果解析成 dict 返回。
这样做的好处是：API 不可能与"被验证过的那条路"产生行为差异 ——
数值、日志、检查点格式都是同一套。

还没做（会在文档里标注）：把模型定义与训练循环搬进包内、改成进程内调用。
那时 API 才能摆脱"需要仓库目录"的限制。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

__all__ = ["train", "TrainingResult", "method_supported"]

SUPPORTED = ("sft",)
PLANNED = {
    "pretrain": "预训练入口（同一套循环，换语料格式）—— 计划中",
    "dpo": "需要参考模型 + 偏好对，计划中",
    "grpo": "需要 rollout 采样 + 奖励函数，计划中",
}


class TrainingResult(dict):
    """训练结果。既是 dict（好打印），也能按属性取（好写代码）。"""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None


def _repo_root():
    """包在 <repo>/python/tgai_go/api.py，仓库根是往上两级。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def method_supported(method):
    m = str(method).lower()
    if m in SUPPORTED:
        return True
    hint = PLANNED.get(m, "支持的方法：%s" % ", ".join(SUPPORTED))
    raise NotImplementedError(
        "api.train: 还不支持 method=%r（%s）。修法：先用 method='sft'；"
        "要预训练/DPO/GRPO 请直接调 examples 下的脚本，或等这层补齐" % (method, hint))


def _resolve_tokenizer(weights, tokenizer):
    if tokenizer:
        if not os.path.isfile(tokenizer):
            raise FileNotFoundError(
                "api.train: tokenizer 文件不存在：%s。修法：传 tokenizer.json 的完整路径" % tokenizer)
        return tokenizer
    if weights and os.path.isdir(weights):
        cand = os.path.join(weights, "tokenizer.json")
        if os.path.isfile(cand):
            return cand
    raise ValueError(
        "api.train: 找不到 tokenizer。修法：显式传 tokenizer='.../tokenizer.json'，"
        "或让 weights 指向包含 tokenizer.json 的目录")


def _check_data(data):
    if isinstance(data, (list, tuple)):
        raise NotImplementedError(
            "api.train: 目前只接受 jsonl 路径；传 list/HF 数据集请先用 "
            "tgai_go.train.data.normalize + 写 jsonl")
    if not isinstance(data, str) or not os.path.isfile(data):
        raise FileNotFoundError(
            "api.train: 数据文件不存在：%r。修法：传 jsonl 路径，"
            "每行形如 {\"prompt\": ..., \"completion\": ...}" % (data,))
    return data


def train(weights=None, data=None, method="sft", *, tokenizer=None,
          seq_len=256, steps=300, lr=5e-6, batch=1, limit=2000, holdout=64,
          lr_warmup=40, clip=1.0, out=None, device="cuda", tiny=False,
          quiet=True, timeout=7200, extra_args=None):
    """训练/微调一个模型，返回 `TrainingResult`。

    最小用法（三行）::

        from tgai_go import api
        r = api.train(weights="Qwen2.5-0.5B", data="my.jsonl")
        print(r.probe_final, r.verdict)

    参数
    ----
    weights : safetensors 目录（含 model.safetensors + tokenizer.json）
    data    : jsonl 路径，每行 ``{"prompt":…, "completion":…}``
    method  : 目前只支持 ``"sft"``；pretrain/dpo/grpo 会明确报"计划中"
    tiny    : True 时用随机迷你权重跑（CPU 冒烟/CI 用，不需要 GPU 与真权重）
    """
    method_supported(method)
    if tiny:
        cmd = [sys.executable, "-X", "utf8", "examples/qwen25_sft.py",
               "--tiny", "--device", "cpu", "--steps", str(steps),
               "--seq-len", str(min(seq_len, 32)), "--warmup", "1"]
        if out:
            cmd += ["--save", out]
        return _run(cmd, quiet=quiet, timeout=timeout)

    if not weights:
        raise ValueError("api.train: 需要 weights（safetensors 目录）。"
                         "没有权重时先用 tiny=True 跑通流程")
    data = _check_data(data)
    tok = _resolve_tokenizer(weights, tokenizer)

    cmd = [sys.executable, "-X", "utf8", "examples/qwen25_sft.py",
           "--safetensors", weights, "--data", data, "--tokenizer", tok,
           "--seq-len", str(int(seq_len)), "--limit", str(int(limit)),
           "--holdout", str(int(holdout)), "--device", device,
           "--warmup", str(int(lr_warmup)), "--clip", repr(float(clip)),
           "--lr", repr(float(lr)), "--batch", str(int(batch)),
           "--steps", str(int(steps))]
    if out:
        cmd += ["--save", out]
    if extra_args:
        cmd += list(extra_args)
    return _run(cmd, quiet=quiet, timeout=timeout)


_PROBE = re.compile(r"([0-9]+\.[0-9]+)\s*[→>-]+\s*([0-9]+\.[0-9]+)")


def _run(cmd, quiet=True, timeout=7200):
    root = _repo_root()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(root, "python") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    text = (proc.stdout or "") + (proc.stderr or "")
    if not quiet:
        print(text)
    return _parse(text, proc.returncode, cmd)


def _parse(text, rc, cmd):
    res = TrainingResult(command=" ".join(cmd), returncode=rc, log=text,
                         probe_base=None, probe_final=None, drop_pct=None,
                         verdict="FAIL", ckpt=None)
    m = _PROBE.search(text)
    if m:
        base, final = float(m.group(1)), float(m.group(2))
        res["probe_base"], res["probe_final"] = base, final
        if base:
            res["drop_pct"] = round((base - final) / base * 100.0, 1)
    if "PASS" in text:
        res["verdict"] = "PASS"
    m2 = re.search(r"已保存.*?[:：]\s*(\S+\.npz)", text)
    if m2:
        res["ckpt"] = m2.group(1)
    return res
