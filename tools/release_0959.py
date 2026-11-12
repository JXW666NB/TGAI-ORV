# -*- coding: utf-8 -*-
"""0.9.59：C 引擎支持 `.TG` v4（布局与 v3 相同，只放行版本号），并对齐 GO 源码。"""
import glob
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.59"
ENTRY = """## [0.9.59] - 2026-11-12

**C 引擎现在读得 `.TG` v4 —— V2.7 的 v4 权重与 ORV 训练检查点都能加载。**

### 先纠正一个前提：仓库里有**两套**"v4"，别混

| 名字 | 实质 | 谁在用 |
| --- | --- | --- |
| `pack.py::TGWriter.write_stream(version=4)` | **v3 布局 + 头里 version=4**（`magic=0x49414754`） | ORV 训练检查点、V2.7 的 `TGAI_NB_V2.7_v4.tg` |
| `tg4.py` | 真正的新容器：`magic=0x00344754` + **section table** + `NAM` 名字表 | **全仓库无人 import**（写了没启用）|

实测证据（V100）：`TGAI_NB_V2.7_v4.tg`（6.06 GB）与 `ckpt_test/step_000002/weights.tg`
都是 `magic=0x49414754` + `version=4`，且 Python 的 `read_tg()` 能直接解析出
`layers=28 d_model=1024 n_heads=16` 并读出张量表 —— 也就是说**布局与 v3 完全一致**。

### 所以引擎侧只需放行版本号

`src/tgai_format.h` 新增 `TG_VERSION_V4` 与 `TG_IS_V3FAMILY(v)`；
`src/tgai_engine.c` 三处版本判定改用该宏（扩展段解析、版本校验、v3 专属校验），
报错文案改为 `版本不兼容 (支持 v1/v3/v4)`。**没有新增读取路径** —— 因为不需要。

### 实测（V100，32GB）

| 文件 | 版本 | 结果 |
| --- | --- | --- |
| `TGAI_NB_V2.7_v4.tg` | 4 | ✅ `layers=28 d_model=1024 vocab=32779 max_seq=8192` |
| `TGAI_NB_V2.7.tg` | 3 | ✅ 同上（向后兼容）|

端到端出字（不只是加载）：

```
encode('你好') -> [2, 911, 2002]
generate -> '同管航,。450**更版取合的佾\\n很or'
```

### 引擎源码与 GO 对齐

逐文件比对 `src/`：只有 `tgai_engine.c` 与 `tgai_engine.h` 落后于 GO，已同步；
其余 18 个文件逐字节相同。

### 新增工具

* `tools/set_tg_version.py` —— 只改头里的版本号（v3↔v4），并**拒绝** TG4 新容器；
* `tools/probe_tg_layout.py` —— 一句话分辨"v3 家族 / TG4 新容器 / 未知"，并试跑 `read_tg`；
* `tools/gen_smoke.py` —— 加载 → 编码 → 生成 → 解码的端到端冒烟（"能加载"与"能出字"分开验）；
* `build_engine.bat` —— 从**本仓库** `src/` 构建（旧的 `rebuild_cpu.bat` 写死编旧 GO 树）。

### 仍未做

`tg4.py` 那套 TG4 容器（section table）目前**没有任何读写方使用**，
本版没有把它接进推理链路 —— 需要单独立项，不能与本次的 v4 混为一谈。

套件：673 passed / 42 skipped（新增 2 例 v4 回归）。
"""


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


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

    # 把刚编好的 DLL 装进包里（构建产物在 cpu/build/）
    built = os.path.join(ROOT, "cpu", "build", "libtgai_engine.dll")
    dst = os.path.join(ROOT, "python", "tgai_go", "_native", "libtgai_engine.dll")
    if os.path.isfile(built):
        shutil.copy2(built, dst)
        print("已装入:", dst, os.path.getsize(dst), "字节")
    want = {"libtgai_engine.dll": sha(dst),
            "libomp.dll": sha(os.path.join(ROOT, "python", "tgai_go", "_native", "libomp.dll")),
            "libtgai_engine_vk.dll": sha(os.path.join(
                ROOT, "python", "tgai_go", "_native", "libtgai_engine_vk.dll"))}

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
                for k, h in sorted(want.items()):
                    n = [x for x in z.namelist() if x.endswith("/_native/" + k)]
                    got = hashlib.sha256(z.read(n[0])).hexdigest()[:12] if n else None
                    hit = (got == h[:12])
                    print("   %-22s %s (%s)" % (k, hit, got))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
