# -*- coding: utf-8 -*-
"""把工作目录里的「白名单文件」同步到真正的 TGAI-GO 仓库并推送。

为什么需要它：2026-10-30 我在朋友的展示备份仓库（Shuyuan-Industrial-Engine）上推了一整天，
TGAI-GO 一个字没更新。原因是两个仓库长得像、只差 remote 指向，靠记忆容易错。
所以把"往哪推、推什么"固化成脚本，**只允许**这几个路径进公开仓库。

用法：
    python tools/sync_to_tgai_go.py --dry-run      # 只看会改哪些文件
    python tools/sync_to_tgai_go.py -m "提交说明"   # 真推

约定：
* 克隆/缓存目录默认 `_tgai_go_real`（相对工作区根），没有就自动 clone；
* 只复制 WHITELIST 里的路径，**不删除**真仓库里别人已有的文件；
* 绝不 force push；推送前会打印 `git diff --stat` 供人确认。
"""
import argparse
import os
import shutil
import subprocess
import sys

# 工作区根（本脚本在 <work>/_repos/Shuyuan-Industrial-Engine/tools/ 下，或 <work>/tools/ 下）
_HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if os.path.basename(WORK) == "" or not os.path.isdir(os.path.join(WORK, "_repos")):
    WORK = os.path.dirname(os.path.dirname(_HERE))
SRC_ROOT = os.path.join(WORK, "_repos", "Shuyuan-Industrial-Engine")
CLONE = os.path.join(WORK, "_tgai_go_real")
REMOTE = "https://github.com/JXW666NB/TGAI-GO.git"

# 允许进公开仓库的路径（相对 SRC_ROOT）。新增文件请显式登记，避免把实验垃圾推上去。
WHITELIST = [
    "python/tgai_go/train/__init__.py",
    "python/tgai_go/train/cuda.py",
    "python/tgai_go/train/backend.py",
    "python/tgai_go/train/dist.py",
    "python/tgai_go/train/doctor.py",
    "python/tgai_go/train/export.py",
    "python/tgai_go/train/pipeline.py",
    "python/tgai_go/train/bpe.py",
    "docs/PRODUCTION_READINESS.md",
    "docs/QUICKSTART_TRAIN.md",
    "tools/accept.py",
    "tools/soak_test.py",
    "tools/profile_step.py",
    "examples/exam2_chat10m_tgai.py",
    "examples/make_twins.py",
    "examples/stream_data.py",
    "scripts/",
    "tools/sync_to_tgai_go.py",
    # ★ CUDA 算子源码（2026-10-30 漏推，用户发现）：训练引擎运行时用
    #   attn_nop_source_path() 从 training/kernels/ 读 tgai_attn_nop.cu 单独编译，
    #   其余算子源码是「原始出处」，也必须公开。
    "training/kernels/",
    "python/tgai_go/train/kernels/",
]

# ★ 全部 C/CUDA 源码树（用户要求：所有算子、所有 C 代码都开源）
#   只收源码与头文件，跳过 third_party（上游 vendored）、build、__pycache__。
SOURCE_TREES = [
    "src/", "native/", "cuda/", "cpu/", "embedded/", "shaders/", "tcc/",
    "training/", "python/tgai_go/",
]
SRC_EXT = {".c", ".h", ".cu", ".cuh", ".cpp", ".hpp", ".cc", ".hh",
           ".s", ".S", ".asm", ".inc", ".def", ".rc", ".py"}
SKIP_DIR = ("third_party", "build", "__pycache__", ".git", "node_modules", "dist")

PROXY = os.environ.get("TGAI_GIT_PROXY", "http://127.0.0.1:7890")


def run(cmd, cwd, check=True, capture=True):
    kw = dict(cwd=cwd, text=True, encoding="utf-8", errors="replace")
    if capture:
        kw.update(capture_output=True)
    return subprocess.run(cmd, check=False, **kw)


def git(args, cwd, check=True):
    r = run(["git"] + args, cwd)
    if check and r.returncode != 0:
        raise SystemExit("git %s 失败：\n%s\n%s" % (" ".join(args), r.stdout, r.stderr))
    return r


def ensure_clone():
    if os.path.isdir(os.path.join(CLONE, ".git")):
        return
    print("克隆 %s -> %s" % (REMOTE, CLONE))
    r = run(["git", "-c", "http.proxy=" + PROXY, "-c", "https.proxy=" + PROXY,
             "clone", REMOTE, CLONE], WORK)
    if r.returncode != 0:
        raise SystemExit("克隆失败：\n%s\n%s" % (r.stdout, r.stderr))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-m", "--message", default="")
    a = ap.parse_args(argv)

    if not os.path.isdir(SRC_ROOT):
        raise SystemExit("找不到工作仓库：%s" % SRC_ROOT)
    ensure_clone()

    copied = []

    def _copy_one(rel):
        s = os.path.join(SRC_ROOT, *rel.split("/"))
        d = os.path.join(CLONE, *rel.split("/"))
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if os.path.exists(d) and open(s, "rb").read() == open(d, "rb").read():
            return False
        if a.dry_run:
            print("  将更新：%s" % rel)
        else:
            shutil.copyfile(s, d)
            print("  已更新：%s" % rel)
        return True

    for rel in WHITELIST:
        s = os.path.join(SRC_ROOT, *rel.split("/"))
        if not os.path.exists(s):
            print("  跳过（源不存在）：%s" % rel)
            continue
        if rel.endswith("/"):                      # 目录：整目录逐个文件同步
            for root, _dirs, files in os.walk(s):
                for fn in sorted(files):
                    if fn.endswith((".pyc", ".pyo")):
                        continue
                    sub = os.path.relpath(os.path.join(root, fn), SRC_ROOT)
                    if _copy_one(sub.replace(os.sep, "/")):
                        copied.append(sub)
            continue
        if _copy_one(rel):
            copied.append(rel)
    if False:
        pass

    # ---- 源码树全量 ----
    n_src = 0
    for tree in SOURCE_TREES:
        base = os.path.join(SRC_ROOT, *tree.rstrip("/").split("/"))
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP_DIR]
            for fn in sorted(files):
                ext = os.path.splitext(fn)[1]
                if ext not in SRC_EXT:
                    continue
                sub = os.path.relpath(os.path.join(root, fn), SRC_ROOT)
                if sub.replace(os.sep, "/") in copied:
                    continue
                if _copy_one(sub.replace(os.sep, "/")):
                    copied.append(sub)
                    n_src += 1
    print("  其中源码树新增/更新 %d 个" % n_src)

    print("\n改动文件 %d 个" % len(copied))
    if not copied:
        print("没有需要同步的内容。")
        return 0
    if a.dry_run:
        print("（dry-run，未写入、未提交）")
        return 0

    git(["add", "-A"], CLONE)
    st = git(["--no-pager", "diff", "--cached", "--stat"], CLONE).stdout
    print("\n" + (st or "（空）"))
    msg = a.message or ("sync: " + ", ".join(os.path.basename(p) for p in copied[:4])
                        + (" 等" if len(copied) > 4 else ""))
    git(["-c", "user.name=JXW666NB",
         "-c", "user.email=jxw666nb@users.noreply.github.com",
         "commit", "-q", "-m", msg], CLONE)
    r = git(["-c", "http.proxy=" + PROXY, "-c", "https.proxy=" + PROXY,
             "push", "origin", "main"], CLONE, check=False)
    out = (r.stdout or "") + (r.stderr or "")
    print(out.strip().splitlines()[-1] if out.strip() else "(push 无输出)")
    return 0 if r.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
