# -*- coding: utf-8 -*-
"""0.9.61：文档与 skill 校正（推理用法、多设备能力边界），无代码行为变化。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.61"
ENTRY = """## [0.9.61] - 2026-11-13

**文档校正版：不改代码行为，只把"能做什么/不能做什么"写准。**

### 1. 推理用法写全了（skill §8 重写）

以前那份只写了 `tgai.bat`，而 `tgai.bat` / `tgai_tui.py` / `tgai_chat.py` / `tgai_ui.py`
**都在源码仓库根目录**；`pip install tgai-orv` 装出来的 wheel **没有 console entry point、
也没有这些脚本**，只有 `python -m tgai_go`。现在按安装形态分开写：

| 形态 | 可用命令 |
| --- | --- |
| **只有 pip 包** | `python -m tgai_go gen -m m.TG "问题"`（单次）、`chat`（交互 REPL，流式，`/exit` `/clear` `/stats`）、`info`、`meta`、`version` |
| **clone 了仓库** | `python tgai_tui.py -m m.TG`（全屏 TUI）、`tgai.bat chat/server/convert/bench/rebuild/selfcheck/help` |
| **写在 Python 里** | `tgai_go.load()` → `generate` / `session.stream` / `encode` / `decode` |

并补充：**训练检查点 `weights.tg` 可直接当模型推理**（0.9.60 起是 TG4，自带分词表），
不必先导出 —— V100 上实测 `layers=28 d_model=1024 vocab=32779`，`encode`/`generate` 正常。

同时删掉了三处**不存在**的说法：`tgai.bat models`（无此子命令）、
`libtgai_engine_gpubridge.dll`（构建产物里没有这个文件）、以及把 TUI 斜杠命令列成一长串
（实际只核实到 `/help` `/params` `/exit`）。

### 2. 多设备能力边界改准（skill §2.2）

原文把"TG Link（显卡间 P2P / 多厂商交火）"和"进程内混用后端"列进了能力表，
容易被读成"ORV 有这条路"。**两者都只是文档里的设想：未实现、未验证。**
现在整节改成一句话结论 —— **ORV 目前只保证单卡**，并明确列出三种"常被误以为有"的东西
（TG Link、进程内混后端、同后端多卡同进程）都不存在；`dist.py` 只是多进程、且未在真多卡验证。
§11 参考表里那条也从"权威"改成"只是路线设想，别据此承诺能力"。

### 3. `docs/TRAIN_V27_ORV.md` §19 的"待确认项"清零

原文写着"待确认：引擎当前能否读 v4"。现在已经落地并实测，改为记录**结果**：
检查点与导出写 TG4、引擎有 `tg4_attach()` 分支、`read_tg` 按 magic 分派、
v3 与"v3 布局 + version=4"仍可读、迁移工具 `tools/to_tg4.py`，
以及"落地前真实 TG4 文件为 0 个、故无迁移负担"。

### 4. 顺带修正的过时说明

skill 里"`.TG` 头里没有 head_dim 字段"改成"**v3 容器**没有；**TG4 头里有独立 `head_dim`**（引擎会用它），
但 `save_tg` 目前仍对非自然 head_dim 报错 —— 要放开需先在目标模型上实测"。

套件：682 passed / 42 skipped。
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
        print(r.stdout[-800:], r.stderr[-800:])
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
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
