# -*- coding: utf-8 -*-
"""升级版本 → 更新 CHANGELOG → 构建 sdist/wheel → 校验 wheel 内含今天的修复。

用法：python tools/bump_and_build.py [--version 0.7.0] [--publish]
  --publish 需要 twine 且配置好凭据（本机当前没有凭据，会明确报错而不是半途失败）。
"""
import argparse
import glob
import io
import os
import re
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYPROJECT = os.path.join(ROOT, "pyproject.toml")
CHANGELOG = os.path.join(ROOT, "CHANGELOG.md")

ENTRY = """## [{ver}] - {date}

### 修复（与 PyTorch 数值对齐，实测 15 步 loss 相对偏差 1.00e-05）

- **注意力缺因果掩码**：`v27_train.py::fwd` 原为 `softmax((q@k.T)*scale)`，**没有 mask**，
  等价于双向注意力（能看到未来 token）。已加上三角 −1e9 掩码。这是语义级 bug：
  修复前单层 loss 与 PyTorch 差 2.56e-02，修复后 1.35e-07。
- **CUDA MoE 的 softmax 作用域错误**：`_combine_cuda` 在**全部专家**上做 softmax 再取 top-k，
  而语义（CPU 路径 / 引擎 `tgai_cuda.cu:301-307` / PyTorch）是**只在选中的 k 个上** softmax。
  现改用 `moe_topk` 返回的选中值 `val_buf` 直接 softmax。实测 scale=8 时 4.55e-02 → 1.42e-06。
- **训练路径丢失输入 embedding 梯度**（tied head）：训练喂 `x0` 用的是 numpy 常量，
  tied embedding 只拿到"输出投影"一侧梯度（实测差 34%）。新增
  `V27Trainer.add_embed_grad()`，`backward_and_step` 记录 `last_input_grad`，
  用 `rt.emb_bwd` 把输入侧梯度散射回 `head.weight.grad`。
- **`emb_bwd` 的 id dtype**：传 int64 会被内核按 int32 解读，梯度落到序列里不存在的行
  （实测 row 0 拿到 0.62，而该 token 根本不在输入里）。改为 int32 后总梯度误差 9.29e-01 → 5.85e-06。

### 新增

- `tgai_go.train.v27_train.V27Trainer.add_embed_grad(ids, g_x0=None)`：补齐 tied embedding 的输入侧梯度。
- `V27Trainer.last_input_grad`：逐层反向结束后保留 `d(loss)/d(x0)`。

### 对齐证据（可复现）

前向 1/2/4 层 loss 相对误差 1.35e-07 / 6.65e-08 / 0.00e+00；分组梯度 ≤7e-6；
tied-embed 总梯度 5.85e-06；AdamW 单步参数 ≤5.09e-08；逐层重算反向 vs 图反向 ≤3.71e-07；
ORV 自比噪声底（双进程）3.20e-07；真实语料 15 步曲线最差 1.00e-05。
全 28 层前向 loss：ORV 1.3444903 vs PyTorch 1.3444886（相对 1.2e-06）。

> 注意：`attention` 由双向改为因果会**改变已有模型的训练/推理行为**，
> 之前用旧版训练的 checkpoint 语义上等价于另一个模型。故按 minor 版本升级。

"""


def read_text(p):
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with io.open(p, encoding=enc) as f:
                return f.read(), enc
        except UnicodeDecodeError:
            continue
    with io.open(p, encoding="utf-8", errors="replace") as f:
        return f.read(), "utf-8(replace)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="0.7.0")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    s, enc = read_text(PYPROJECT)
    m = re.search(r'^version\s*=\s*"([^"]+)"', s, re.M)
    if not m:
        print("!! pyproject 里找不到 version 行")
        return 2
    old = m.group(1)
    s2 = re.sub(r'^version\s*=\s*"[^"]+"', 'version = "%s"' % args.version, s, count=1, flags=re.M)
    with io.open(PYPROJECT, "w", encoding="utf-8", newline="\n") as f:
        f.write(s2)
    print("版本：%s -> %s（pyproject）" % (old, args.version))

    cl, cenc = read_text(CHANGELOG)
    if "[%s]" % args.version in cl:
        print("CHANGELOG 已有该版本，跳过")
    else:
        import datetime
        entry = ENTRY.format(ver=args.version, date=datetime.date.today().isoformat())
        # 插到第一个 "## [" 标题之前（最新在上）
        idx = cl.find("## [")
        new = (entry + cl) if idx < 0 else (cl[:idx] + entry + cl[idx:])
        with io.open(CHANGELOG, "w", encoding="utf-8", newline="\n") as f:
            f.write(new)
        print("CHANGELOG 已插入 %s 条目（原编码 %s）" % (args.version, cenc))

    for d in ("dist", "build"):
        p = os.path.join(ROOT, d)
        if os.path.isdir(p):
            import shutil
            shutil.rmtree(p)
            print("清理 %s/" % d)
    for f in glob.glob(os.path.join(ROOT, "python", "*.egg-info")):
        import shutil
        shutil.rmtree(f, ignore_errors=True)

    r = subprocess.run([sys.executable, "-m", "build", "--outdir", "dist"], cwd=ROOT,
                       capture_output=True, text=True)
    print("build rc=%d" % r.returncode)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        return r.returncode

    wheels = glob.glob(os.path.join(ROOT, "dist", "*.whl"))
    sdists = glob.glob(os.path.join(ROOT, "dist", "*.tar.gz"))
    print("产物：%s" % [os.path.basename(x) for x in wheels + sdists])

    ok = True
    for w in wheels:
        with zipfile.ZipFile(w) as z:
            names = z.namelist()
            meta = [n for n in names if n.endswith("METADATA")]
            ver_ok = False
            if meta:
                txt = z.read(meta[0]).decode("utf-8", "replace")
                ver_ok = ("Version: %s" % args.version) in txt
            checks = {}
            for key, needle in (("因果掩码", b"_causal_mask"),
                                ("embedding 梯度", b"add_embed_grad"),
                                ("MoE 选中 softmax", b"bw_sel_topk")):
                hit = False
                for n in names:
                    if n.startswith("tgai_go/train/") and n.endswith(".py"):
                        if needle in z.read(n):
                            hit = True
                            break
                checks[key] = hit
            print("wheel %s: 版本正确=%s" % (os.path.basename(w), ver_ok))
            for k, v in checks.items():
                print("   %-16s 包含=%s" % (k, v))
                ok = ok and v
            ok = ok and ver_ok

    if args.publish:
        if not wheels:
            print("!! 没有 wheel")
            return 3
        r = subprocess.run([sys.executable, "-m", "twine", "upload"] + wheels + sdists,
                           cwd=ROOT, capture_output=True, text=True)
        print("twine rc=%d" % r.returncode)
        print((r.stdout or "")[-1500:])
        print((r.stderr or "")[-1500:])
        ok = ok and r.returncode == 0
    else:
        print("未上传（缺凭据）。配置好后执行：")
        print("  $env:TWINE_USERNAME='__token__'; $env:TWINE_PASSWORD='pypi-...'")
        print("  python -m twine upload dist/*")

    print("校验结果：%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
