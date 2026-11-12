# -*- coding: utf-8 -*-
"""0.9.60：检查点与导出改用 **TG4**；`read_tg` 透明读 TG4。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.60"
ENTRY = """## [0.9.60] - 2026-11-13

**检查点与导出改用 TG4 容器**，兑现「以后统一 v4」；`read_tg` 现在两种容器都认。

### 为什么换容器

此前"v4"其实是 **v3 布局 + 头里 version=4**：权重表只有 FNV 哈希、**没有名字**，
于是按名映射 / 部分加载 / 按名 diff 全都做不了。TG4 的 `NAM` 直接给出名字，
头里还有独立的 `head_dim` 字段。

### 改了什么

| 位置 | 改动 |
| --- | --- |
| `ckpt.save_checkpoint` | 权重写 **TG4**（`tg4.write_tg4_stream`，流式、峰值=单张量），**分词表随权重**；优化器仍逐层 npz |
| `ckpt.CKPT_WEIGHTS` | **文件名保持 `weights.tg`** —— 路径是外部契约，容器由 magic 区分，旧检查点仍可读 |
| `export.save_tg` | 导出改 TG4；`TGWriter` 只用于编码分词段（TOK 布局与 v3 一致，不重复实现） |
| `read_tg` | **透明读 TG4**：`_read_tg4()` 把 NAM 翻成同构 `info`（`table`/`data_off`/`tokens` 语义对齐 v3）⇒ `load_tensor` / `list_tensors` / `ckpt.load_tg` / `v27_train` **全部无需改动** |
| `list_tensors` | TG4 自带名字时直接用文件里的名字，不再依赖旁挂 |

### 修掉一个自己引入的 bug

`_read_tg4` 起初只读**头里的** `tok_off/tok_len`，而 `write_tg4_stream` 把 TOK 放在
**分区表**里（头里留 0）⇒ 读出来"分词表为空"。已改为**头优先、空则回退到 `TOK` 分区**。
这个 bug 是被"检查点写出的 TG4 应自带分词表"这条测试抓到的。

### 验证

| 项 | 结果 |
| --- | --- |
| 本地套件 | **681 passed / 42 skipped** |
| V100 GPU 门 | **287 passed / 12 skipped / 0 failed** |
| 引擎读真实 TG4 | `TGAI_NB_V2.7.tg4`（982 张量 / 6.0 GB）→ `layers=28 d_model=1024 vocab=32779`，`encode`/`generate` 正常 |
| 检查点 → 引擎 | `save_checkpoint` 写出的 TG4 **直接被引擎加载**（组合用例，非只验两半） |
| 导出 | `save_tg` 输出 magic=TG4、名字可枚举、分词表随文件 |

### 新增工具

* `tools/to_tg4.py` —— v3 → TG4 流式转换（需名字边车；峰值内存=单张量）；
* `tools/ls_dirs.py` —— 列目录内容（远端排查用，避免 shell 里拼 `&`）。

### 兼容性

* **v3 与"v3 布局 + version=4"仍可读**（`read_tg` 按 magic 分派；引擎侧 `TG_IS_V3FAMILY`）；
* 旧检查点目录（`weights.tg` 是 v3 家族）继续可续训 —— 只读路径不变；
* `TG4` 容器**没有**历史文件（落地前已扫盘确认为 0 个），本版起它才是"新写出的 v4"。
"""


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def main():
    write(os.path.join(ROOT, "pyproject.toml"),
          re.sub(r'(?m)^version\s*=\s*"[^"]+"', 'version = "%s"' % VER,
                 read(os.path.join(ROOT, "pyproject.toml")), count=1))
    ini = os.path.join(ROOT, "python", "tgai_go", "__init__.py")
    write(ini, re.sub(r'(?m)^(\s*__version__\s*=\s*")[^"]+(")',
                      r"\g<1>%s\g<2>" % VER, read(ini)))
    print("版本:", re.findall(r'(?m)^\s*__version__\s*=\s*"([^"]+)"', read(ini)))

    cl = os.path.join(ROOT, "CHANGELOG.md")
    txt = read(cl)
    if "[%s]" % VER not in txt:
        i = txt.find("## [")
        write(cl, (ENTRY + "\n" + txt) if i < 0 else (txt[:i] + ENTRY + "\n" + txt[i:]))
        print("CHANGELOG OK")

    for name in ("dist", "build"):
        q = os.path.join(ROOT, name)
        if os.path.isdir(q):
            shutil.rmtree(q)
    for f in glob.glob(os.path.join(ROOT, "python", "*.egg-info")):
        shutil.rmtree(f, ignore_errors=True)
    r = subprocess.run([sys.executable, "-m", "build", "--outdir", "dist"], cwd=ROOT,
                       capture_output=True, text=True)
    print("build rc=%d" % r.returncode)
    if r.returncode:
        print(r.stdout[-600:], r.stderr[-600:])
        return r.returncode
    pr = os.path.join(ROOT, "publish_ready")
    if os.path.isdir(pr):
        shutil.rmtree(pr)
    os.makedirs(pr)
    files = glob.glob(os.path.join(ROOT, "dist", "*"))
    for f in files:
        shutil.copy2(f, pr)
    ok = True
    for f in files:
        if f.endswith(".whl"):
            with zipfile.ZipFile(f) as z:
                meta = [n for n in z.namelist() if n.endswith("METADATA")][0]
                ok = ok and (("Version: %s" % VER) in z.read(meta).decode("utf-8", "replace"))
                rt = z.read("tgai_go/read_tg.py").decode("utf-8", "replace")
                ck = z.read("tgai_go/train/ckpt.py").decode("utf-8", "replace")
                ex = z.read("tgai_go/train/export.py").decode("utf-8", "replace")
                checks = {
                    "read_tg 认 TG4": "_read_tg4" in rt and "TG4_MAGIC" in rt,
                    "TOK 回退分区": "tok_len < 8 and 1 in secs" in rt,
                    "检查点写 TG4": "write_tg4_stream" in ck,
                    "导出写 TG4": "write_tg4_stream" in ex,
                }
                for kk, hit in checks.items():
                    print("   %-16s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
