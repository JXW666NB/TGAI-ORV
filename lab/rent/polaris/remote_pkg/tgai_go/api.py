# -*- coding: utf-8 -*-
"""十行代码训练的入口：`from tgai_go import api; api.train(...)`。

设计取舍（如实说明）
--------------------
模型与训练循环住在 `tgai_go.train.qwen` 里（**与 examples/qwen25_sft.py 共用同一份代码**，
不是复制）。所以这里是**进程内**调用：不需要仓库目录，pip 装完即可用。

`engine="subprocess"` 仍保留，用于"想复现 CLI 行为的对照"场景。
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import sys

import numpy as np

__all__ = ["train", "TrainingResult", "method_supported"]

SUPPORTED = ("sft", "pretrain")
PLANNED = {
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


def train(weights=None, data=None, method="sft", *, corpus=None, cfg=None,
          tokenizer=None,
          seq_len=256, steps=300, lr=5e-6, batch=1, limit=2000, holdout=64,
          lr_warmup=40, clip=1.0, out=None, device="cuda", tiny=False,
          quiet=True, timeout=7200, extra_args=None, engine="inproc"):
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
    # ★ 参数校验必须在**分派之前**：否则进程内路径会先尝试建 CUDA 上下文，
    #   在没卡的机器上抛出"找不到 nvcuda.dll"，把真正的问题（路径写错）盖掉。
    if str(method).lower() == "pretrain":
        if not corpus:
            raise ValueError("api.train: method='pretrain' 需要 corpus（.txt 或每行 {\"text\"}）。"
                             "修法：传 corpus='my_corpus.txt'")
        tokenizer = _resolve_tokenizer(None, tokenizer)
    elif not tiny:
        if not weights:
            raise ValueError("api.train: 需要 weights（safetensors 目录）。"
                             "没有权重时先用 tiny=True 跑通流程")
        data = _check_data(data)
        tokenizer = _resolve_tokenizer(weights, tokenizer)

    if method_supported(method) and str(method).lower() == "pretrain":
        return _train_pretrain_inproc(corpus, cfg, tokenizer=tokenizer, seq_len=seq_len,
                                      steps=steps, lr=lr, batch=batch, limit=limit,
                                      holdout=holdout, warmup=lr_warmup, clip=clip,
                                      out=out, device=device, quiet=quiet, tiny=tiny)

    if engine != "subprocess":
        return _train_inproc(weights, data, tokenizer, seq_len=seq_len, steps=steps,
                             lr=lr, batch=batch, limit=limit, holdout=holdout,
                             warmup=lr_warmup, clip=clip, out=out, device=device,
                             tiny=tiny, quiet=quiet)

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


def _train_pretrain_inproc(corpus, cfg, *, tokenizer, seq_len, steps, lr, batch,
                           limit, holdout, warmup, clip, out, device, quiet=True,
                           tiny=False):
    """从零预训练：随机初始化 -> 纯文本定长分块 -> 全 token LM loss。"""
    import contextlib

    from .train import qwen as Q

    T = __import__("tgai_go.train", fromlist=["train"])
    T.set_default_device("cpu" if tiny else device)

    if not cfg:
        raise ValueError("api.train: method='pretrain' 需要 cfg（如 V/D/L/NH/KV/FF/DH）。"
                         "修法：传 cfg=dict(V=32000, D=256, L=4, NH=8, KV=2, FF=688, DH=32)")

    model = Q.Qwen25(cfg, Q.make_random_weights(cfg))
    seqs = Q.build_lm_sequences(Q.load_tokenizer(tokenizer), corpus, int(seq_len),
                                int(limit) if limit else None)
    if len(seqs) < 2:
        raise ValueError(
            "api.train: 语料太小，切不出足够的训练块（只有 %d 块，每块 %d token）。"
            "修法：加长语料或减小 seq_len" % (len(seqs), int(seq_len)))

    rng = np.random.default_rng(0)
    idx = list(range(len(seqs)))
    rng.shuffle(idx)
    h = max(1, min(int(holdout), len(seqs) // 4))
    hold = set(idx[:h])
    probes = [seqs[i] for i in sorted(hold)][:8]
    train_seqs = [s for i, s in enumerate(seqs) if i not in hold]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf if quiet else contextlib.nullcontext()):
        base = Q._probe_multi(model, probes, int(seq_len))
        Q.sft(model, train_seqs, int(steps), float(lr), int(seq_len), probe=probes,
              clip=float(clip), warmup=int(warmup), batch=int(batch))
        final = Q._probe_multi(model, probes, int(seq_len))

    ckpt = None
    if out:
        Q.save_params(model, out)
        ckpt = out
    drop = round((base - final) / base * 100.0, 1) if base else None
    return TrainingResult(command="in-process(pretrain)", returncode=0,
                          log=buf.getvalue(), probe_base=base, probe_final=final,
                          drop_pct=drop,
                          verdict="PASS" if (drop is not None and drop >= 5.0) else "FAIL",
                          ckpt=ckpt, engine="in-process", n_seqs=len(seqs))


def _train_inproc(weights, data, tok_path, *, seq_len, steps, lr, batch, limit,
                  holdout, warmup, clip, out, device, tiny, quiet=True):
    """进程内训练：建模型 -> 造序列 -> 切留出集 -> 跑 sft()。"""
    from .train import qwen as Q

    T = __import__("tgai_go.train", fromlist=["train"])
    T.set_default_device("cpu" if tiny else device)

    if tiny:
        cfg = dict(Q.TINY_CFG)
        model = Q.Qwen25(cfg, Q.make_tiny_weights(cfg))
        seqs = Q.tiny_sequences(cfg, n=8, seq_len=min(int(seq_len), 32))
        probes = [s[:min(len(s), 24)] for s in seqs[:4]]
        steps = max(1, int(steps))
        warmup = min(int(warmup), steps)
    else:
        ck = Q.load_safetensors_weights and None          # 仅为可读性占位
        from tgai_go.train import checkpoints as C
        ck = C.load_safetensors_dir(weights)
        cfg = Q.cfg_from_checkpoint(ck)
        model = Q.Qwen25(cfg, dict(ck.weights))
        del ck
        tok = Q.load_tokenizer(tok_path)
        seqs = Q.build_sequences(tok, data, int(seq_len), int(limit))
        rng = np.random.default_rng(0)
        idx = list(range(len(seqs)))
        rng.shuffle(idx)
        hold = set(idx[:max(1, int(holdout))])
        probes = [seqs[i] for i in sorted(hold)][:8]
        seqs = [s for i, s in enumerate(seqs) if i not in hold]

    import contextlib
    buf = io.StringIO()
    sink = buf if quiet else contextlib.nullcontext()
    with contextlib.redirect_stdout(sink):
        base = Q._probe_multi(model, probes, int(seq_len))
        Q.sft(model, seqs, int(steps), float(lr), int(seq_len), probe=probes,
              clip=float(clip), warmup=int(warmup), batch=int(batch))
        final = Q._probe_multi(model, probes, int(seq_len))
    ckpt = None
    if out:
        Q.save_params(model, out)
        ckpt = out
    drop = round((base - final) / base * 100.0, 1) if base else None
    return TrainingResult(command="in-process", returncode=0, log=buf.getvalue(),
                          probe_base=base, probe_final=final, drop_pct=drop,
                          verdict="PASS" if (drop is not None and drop >= 5.0) else "FAIL",
                          ckpt=ckpt, engine="in-process")


_PAIR = re.compile(r"([0-9]+\.[0-9]+)\s*[→>-]+\s*([0-9]+\.[0-9]+)")
# ★ 只认"探针/probe"那一行：示例里同时会打印
#   「训练 loss（轮转样本，不可比）：A → B」和「探针 loss（固定序列，可比）：C → D」，
#   取第一个匹配会拿到**不可比**的那个数（实测踩过：API 报 5.5995→5.541，
#   而真正可比的探针是 5.5789→5.2458）。
_PROBE_LINE = re.compile(r"(探针|probe)", re.I)


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
    m = None
    for line in text.splitlines():
        if _PROBE_LINE.search(line):
            m = _PAIR.search(line)
            if m:
                break
    if m is None:                      # 兜底：没有任何"探针"标记时才用第一对
        m = _PAIR.search(text)
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
