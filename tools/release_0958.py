# -*- coding: utf-8 -*-
"""0.9.58：修复随包原生库损坏，推理在 .TG v3 上真正可用；构建改为仓库相对。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.58"
ENTRY = """## [0.9.58] - 2026-11-12

**修复随包原生库损坏 —— 推理在 `.TG` v3 上真正可用。** 我此前的判断有误，这里更正。

### 真因：不是"需要重编"，而是仓库里三个 DLL 都被截断/错位

外部报告指出两个引擎 DLL 的 PE 头错位 1 字节，我据此认为必须重编。实际排查发现：

| 文件 | 仓库里的 | 本机另一个干净副本 | 结论 |
| --- | --- | --- | --- |
| `libtgai_engine.dll` | PE 签名在 `e_lfanew` 前 1 字节 | 297472 字节，PE 正常 | **损坏** |
| `libtgai_engine_vk.dll` | 同上（0x108 vs 0x107）| 331776 字节，PE 正常 | **损坏** |
| **`libomp.dll`** | **968175 字节** | **968192 字节** | **少 17 字节，损坏** |

`libomp.dll` 这一条是**之前没人发现的**：它 PE 头正常，所以只查 PE 会被漏掉；
但引擎在 `DllMain` 阶段就要绑定 OpenMP 符号，坏的 libomp 会让
`ctypes.CDLL("libtgai_engine.dll")` 直接**访问违例**（`0xC0000005`）——
连"已知能用"的引擎 DLL 放进 `_native/` 也会崩，正是这个原因。

三个文件全部换成完好副本后，实测：

```
CDLL OK  libomp.dll
CDLL OK  libtgai_engine.dll
CDLL OK  libtgai_engine_vk.dll
[tgai_engine] BUILD 2026-10-18-v3.8-gguf-native-experimental
加载成功: Model layers=6 d_model=192 vocab=8192 max_seq=128
```

### 同时修掉"改了源码却没生效"的构建陷阱

`rebuild_cpu.bat` 里写死 `G=D:\\tgai_nlp\\TGAI GO` —— 它编的是**旧 GO 树**，
不是本仓库的 `src/`。新增 **`build_engine.bat`**：

* 用 `%~dp0` 定位仓库根，`vswhere` 自动找 `vcvars64.bat`（找不到时给出装什么）；
* 编译前断言 `src\\tgai_engine.c` 存在，确保编的是本仓库；
* 实测可用：MSVC 19.44 从本仓库 `src/` 编出 `cpu\\build\\libtgai_engine.dll`。

### 引擎源码已与 GO 对齐

逐文件比对 `src/`：**只有两处不同** —— `tgai_engine.c`（GO 更新）与 `tgai_engine.h`，
已同步进本仓库；其余 18 个文件逐字节相同。

### 新增排查工具

* `tools/pick_engine_dll.py` —— 扫全盘候选 DLL，三重判据（`e_lfanew` 处是否为 `PE\\0\\0`、
  Machine 是否 x64、`CDLL` 是否真能加载），一眼看出哪些可用；
* `tools/probe_infer.py` —— 分离"库能加载"与"模型能读"，并打印 `.TG` 的版本号。

### 仍未解决：`.TG` **V4** 读不了

C 引擎只认 v1/v3（`src/tgai_engine.c:2369`：`"版本不兼容 (支持 v1/v3)"`），
而本仓库的 `tg4.py` 是 **V4** 自描述分区容器、训练检查点就按 v4 写。
所以 **ORV 训出来的 v4 权重目前无法被引擎加载** —— 这是"训推一体"真正缺的一环，
需要给引擎加 v4 解析（分区表 + `NAM` 名字表），不是打包问题。

套件：本地 670 passed / 42 skipped。
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
                nat = {n.split("/_native/")[-1]: z.getinfo(n).file_size
                       for n in z.namelist() if "/_native/" in n and n.endswith(".dll")}
                # 完好副本的字节数（用于挡住"又打进了坏文件"）
                want = {"libomp.dll": 968192, "libtgai_engine.dll": 297472,
                        "libtgai_engine_vk.dll": 331776}
                for k, sz in sorted(want.items()):
                    got = nat.get(k)
                    hit = (got == sz)
                    print("   %-22s %s (期望 %d，实际 %s)" % (k, hit, sz, got))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
