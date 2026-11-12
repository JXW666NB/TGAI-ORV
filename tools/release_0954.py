# -*- coding: utf-8 -*-
"""0.9.54：内核 DH 表达式化（行为不变）+ 文档与 skill 校正。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.54"
ENTRY = """## [0.9.54] - 2026-11-12

**内核不再写死 head 维度**（行为不变，已逐位验证）；同一版校正了文档与 skill 里的事实。

### 内核：去掉对 `DH` 的硬编码

`kernels/tgai_attn_nop.cu` 里只有两处依赖 head 维度，都是写死的数字：

| 原来 | 改为 |
| --- | --- |
| `#define QSTR 72` | `#define QSTR (DH + 8)`（72 = 64 + 8 的补零 stride）|
| 5 处 `smem + 4096` | `smem + (BT * DH)`（4096 = BT×DH）|

**这一步不改变任何行为**，只是为后续支持 dh=96/128 铺路（当前启动器仍只接受 dh=64）。
验证方式：新增 `tools/attn_ref.py`，跑 4 组形状（含 S=100、S=257 这类非 tile 整数倍），
记录前向输出与 q/k/v 梯度共 16 个数组；改动前后**最大绝对差 0.000e+00**。

### 文档：三份实测结论入库

| 文档 | 结论 |
| --- | --- |
| `MEMORY_ATTRIBUTION_2026-11-12.md` | 权重/梯度/优化器两边**字节相同**；差距 100% 在激活。**不开检查点 2.40×，开检查点 1.06×** |
| `ATTENTION_IS_THE_BOTTLENECK_2026-11-12.md` | 分阶段剖析：fp16 的 GEMM 已达 4.4–5.6×，**注意力占 fp16 步时 61% 且 fp16 无收益**；读 PyTorch 源码得出其 tile 表（fp32 仅 `kMaxK=64`，dh=128 需 fp16 + `32×128`）|
| `HARDWARE_INVENTORY.md` | 两台机器的实测能力（含 V100 处于 **TCC**，故 Windows 上 Vulkan 不可用）|

### skill 校正（避免误导）

* `T.from_torch_state_dict` **不存在** → `T.state_dict_from_torch(module)`；
* `T.amp()` 不是"精度开关"：只 cast GEMM 输入，**激活仍 fp32** ⇒ 不省显存、不需要 GradScaler；
* 显存说法改为分区间的实测数字（见上）；
* 补现场已验证架构：**sm_60 P100 / sm_61 GTX 1060 / sm_70 V100**；
* 补 `sdpa_lowmem_ready()` 不接受参数、`.to_cuda()` 不存在、`save(obj, f)` 的参数顺序。

### 其它修复

* CI 此前每次推送都红：`tools/_*.txt` 临时文件带 BOM 已清出索引并加入 gitignore；
  `test_plugin_native.py` 在 Linux 上把 `find_library("c")` 的 soname 当路径用，已改为解析真实路径
  （顺带让这两个用例在 Windows 上从"跳过"变为"真跑"）。

套件：**670 passed / 40 skipped**。
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
                cu = [n for n in z.namelist() if n.endswith("tgai_attn_nop.cu")]
                src = z.read(cu[0]).decode("utf-8", "replace") if cu else ""
                checks = {
                    "cu 随包": bool(cu),
                    "QSTR 表达式": "#define QSTR (DH + 8)" in src,
                    "偏移表达式": "smem + (BT * DH)" in src,
                    "无硬编码 4096": "smem + 4096" not in src,
                }
                for kk, hit in checks.items():
                    print("   %-16s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
