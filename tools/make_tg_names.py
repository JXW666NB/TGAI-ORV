# -*- coding: utf-8 -*-
"""给已存在的 `.TG` 生成**名字旁挂文件** `<model>.TG.names.json`。

为什么需要：`.TG` 权重表只存 FNV-1a 哈希，不存名字。
推理时引擎按预计算哈希查表，够用；但把 `.TG` 当**检查点**时，
"不能枚举参数"就意味着：无法做优化器状态映射、无法按名挂 LoRA、无法部分加载、无法 diff。

本工具从**有名字的源头**（`.pt` / safetensors / GGUF 的键名）反推：
对每个名字算 FNV-1a 64，再与 `.TG` 表项哈希精确匹配（不是猜形状），
匹配上的写名字，匹配不上的标 None —— **绝不编造名字**。

用法：
    python tools/make_tg_names.py --tg D:\\TGAI\\models\\V2.7\\TGAI_NB_V2.7.tg \\
                                  --pt D:\\TGAI\\models\\V2.7\\last_step.pt

    # 只验覆盖率，不写文件
    python tools/make_tg_names.py --tg ... --pt ... --check

退出码：0=全覆盖（或无未命名）  1=有未命中的表项  2=参数/环境错误
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

from tgai_go.read_tg import (close_tg, fnv1a64, list_tensors, read_tg,  # noqa: E402
                             sidecar_path, write_names)


def names_from_pt(path):
    """从 .pt 检查点取张量名（兼容 {model_state_dict:{...}} 与裸 state_dict）。"""
    import torch
    try:
        obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:                      # 老版本 torch 没有 mmap 参数
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for k in ("model_state_dict", "state_dict", "model", "weights"):
            v = obj.get(k)
            if isinstance(v, dict) and v and hasattr(next(iter(v.values())), "shape"):
                return list(v.keys()), {k: v[k].shape for k in v}
    if isinstance(obj, dict) and obj and hasattr(next(iter(obj.values())), "shape"):
        return list(obj.keys()), {k: obj[k].shape for k in obj}
    raise SystemExit("无法从 %s 里找到 state_dict（顶层键：%s）。修法：用 --keys-from 指定一个文本"
                     "文件（每行一个张量名）" % (path, list(obj)[:8] if isinstance(obj, dict) else type(obj)))


def names_from_text(path):
    names = [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()]
    return names, {}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tg", required=True, help="目标 .TG 文件")
    ap.add_argument("--pt", help="有名字的 .pt 检查点")
    ap.add_argument("--keys-from", help="纯文本张量名清单（每行一个），与 --pt 二选一")
    ap.add_argument("--out", help="旁挂输出路径（默认 <tg>.names.json）")
    ap.add_argument("--check", action="store_true", help="只报覆盖率，不写文件")
    a = ap.parse_args(argv)

    if not os.path.isfile(a.tg):
        print("[err] 找不到 .TG：%s" % a.tg)
        return 2
    if bool(a.pt) == bool(a.keys_from):
        print("[err] 必须且只能给 --pt 或 --keys-from 其中一个")
        return 2
    names, shapes = (names_from_pt(a.pt) if a.pt else names_from_text(a.keys_from))
    print("源头张量名 %d 个" % len(names))

    info = read_tg(a.tg)
    try:
        h = info["header"]
        print(".TG: %s (%d B) dtype=%d L=%d D=%d V=%d 表项=%d"
              % (os.path.basename(a.tg), os.path.getsize(a.tg), h["weight_dtype"],
                 h["n_layers"], h["d_model"], h["vocab"], len(info["table"])))
        by_hash = {"%016x" % fnv1a64(n): n for n in names}
        rows = info["table"]
        matched, missing = 0, []
        for e in rows:
            hx = "%016x" % int(e["hash"])
            if hx in by_hash:
                matched += 1
            else:
                missing.append((hx, list(e["shape"]), int(e["dtype"])))
        print("匹配 %d / %d（%.1f%%）" % (matched, len(rows),
                                          100.0 * matched / max(1, len(rows))))
        if missing:
            print("未命中 %d 个，前 5 个（哈希/形状/dtype）：" % len(missing))
            for m in missing[:5]:
                print("   ", m)
        # 源头有、.TG 里没有的（说明导出时被裁剪或改名）
        tg_hashes = {"%016x" % int(e["hash"]) for e in rows}
        src_only = [n for n in names if "%016x" % fnv1a64(n) not in tg_hashes]
        if src_only:
            print("源头有但 .TG 没有：%d 个，前 5：%s" % (len(src_only), src_only[:5]))

        if a.check:
            extra = sum(1 for n in names if n not in ("", None))
            return 0 if matched == len(rows) else 1

        out = write_names(a.tg, info,
                          {"name_by_hash": by_hash,
                           "source": os.path.basename(a.pt or a.keys_from)})
        print("已写：%s" % out)
        # 立刻回读验证：list_tensors 应当能给出名字
        named = [t for t in list_tensors(a.tg) if t[0]]
        print("回读校验：%d / %d 个表项有名字" % (len(named), len(rows)))
        return 0 if len(named) == len(rows) else 1
    finally:
        close_tg(info)


if __name__ == "__main__":
    raise SystemExit(main())
