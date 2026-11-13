# -*- coding: utf-8 -*-
"""0.9.63：训练器支持 fp16 参数存储（2.999B 全参进 32 GB 卡的前提）。"""
import glob, io, os, re, shutil, subprocess, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.63"
ENTRY = """## [0.9.63] - 2026-11-13

**训练器可以把注意力与 MoE 权重按 fp16 存放**，这是 2.999B 全参训练在 32 GB 卡上唯一可行的前提。

### 为什么必须加这个开关

实测（`docs/V27_SFT_BENCH_2026-11-13.md`）：

| 方案 | 权重 | 梯度 | Adam 状态 | 合计 | 32 GB |
| --- | --- | --- | --- | --- | --- |
| fp32 全套（两家默认）| 11.18 | 11.18 | 22.36 | **44.7 GiB** | ✗ |
| fp32 权重/梯度 + fp16 状态 | 11.18 | 11.18 | 11.18 | **33.5 GiB** | ✗ |
| **fp16 权重+梯度+状态** | 5.59 | 5.59 | 11.18 | **22.35 GiB** | ✓ |

所以"换个优化器"不够 —— 权重本身也得降精度。此前 `V27Trainer` 参数是硬建 fp32。

### 改了什么

| 位置 | 改动 |
| --- | --- |
| `Module.load_state_dict_stream(items, dtype=None)` | 设备缓冲**按目标精度直接分配**（`h2d(arr, dtype=...)`），不是先 fp32 再转；默认 `None` = fp32，历史行为不变 |
| `V27Trainer(..., param_dtype="fp32")` | 注意力投影（q/k/v/o）与 MoE 走 `param_dtype`；**RMSNorm / final_norm / RoPE 表 / head 保持 fp32**（数值敏感）|

`Linear` / `MoE` / `RMSNorm` 共用同一个 `load_state_dict_stream`，所以改动只有一处，不是三处。

### 验证与边界

* 本地套件 **699 passed / 42 skipped**（新增 9 条开关语义用例：默认 fp32、fp16/float16/half/FP16 生效、空值/None 回退 fp32）；
* **没有**证明 fp16 数值正确 —— CPU 路径不分配设备缓冲，`dtype` 在那里不生效；数值须在 GPU 上验，测试文件里已注明；
* 预期显存 22.35 GiB（不含激活），seq 512/1024 应可跑，seq 2048 可能贴边 OOM。
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
    write(ini, re.sub(r'(?m)^(\s*__version__\s*=\s*")[^"]+(")', r"\g<1>%s\g<2>" % VER, read(ini)))
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
        print(r.stdout[-700:], r.stderr[-700:]); return r.returncode
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
                t = z.read("tgai_go/train/__init__.py").decode("utf-8", "replace")
                v = z.read("tgai_go/train/v27_train.py").decode("utf-8", "replace")
                for k, hit in {"stream dtype 参数": "def load_state_dict_stream(self, items, dtype=None)" in t,
                               "h2d 带 dtype": "rt.h2d(arr, dtype=_dt)" in t,
                               "trainer param_dtype": 'param_dtype="fp32"' in v,
                               "MoE 走 fp16": "moe_items(), dtype=self._pdt" in v}.items():
                    print("   %-20s %s" % (k, hit)); ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
