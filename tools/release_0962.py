# -*- coding: utf-8 -*-
"""0.9.62：修掉数据并行的 6 个 bug（两个静默、两个卡死）。"""
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VER = "0.9.62"
ENTRY = """## [0.9.62] - 2026-11-13

**数据并行（`tgai_go.train.dist`）修掉 6 个 bug，其中两个静默、两个直接卡死。**
本轮改动集中在 `dist.py`，并在 `cuda.py` 新增 `copy_h2d_into()`。

### 为什么要修（都是实测现象，不是推测）

| # | 现象 | 根因 |
| --- | --- | --- |
| ① | 多卡照常训练、loss 照降，但**各卡初始权重其实不同** | `broadcast()` 读 `p.data`；CUDA 张量的 `.data` 是 `None`，而 `np.ascontiguousarray(None, dtype=float32)` **不报错**、返回 `array([nan])` ⇒ 每个参数被当成"1 个 nan"，权重根本没广播 |
| ② | rank≠0 的权重不会被改写 | 写回目标 `np.ascontiguousarray(...)` 是**临时数组**，不是张量视图 |
| ③ | 一个 rank `rc=0` 正常退出，另一个卡在 `barrier` 直到超时（实测约 3 次挂 1 次） | `close()` 删掉自己落下的 `bar_*` 标记 ⇒ 跑得快的一方把标记删了，慢的一方还在 `_wait_for` |
| ④ | 训练挂死，报错却说"rank0 没创建共享内存" | 按"有梯度的条目"拼缓冲：某 rank 缺一层梯度（MoE 没路由到）时字节总数不同 ⇒ 共享内存**名字**不同 ⇒ 一边创建、一边等 |
| ⑤ | 缺梯度那侧的 `p.grad` 是 **CPU ndarray** | 设备判定看"原来的 grad 是不是张量"，缺梯度时 `orig is None` 走了 `p.grad = val` 兜底 |
| ⑥ | `DataParallel(world=2)` 一构造就阻塞 | 握手写在 `__init__` 里 |

### 改法

* 新增 `_get_data()` / `_set_data()`：按**参数设备**取值/写回；CUDA 走 `to_numpy()` 与新的
  `CudaAPI.copy_h2d_into()`（**原地**写，不新分配 —— `h2d()` 会新分配并漏掉旧缓冲，
  275 MB 的模型每广播一次就漏一份）；
* `all_reduce_grads()`：按 `params` 固定顺序遍历，**缺梯度按 0 补齐**，保证各 rank 字节数一致；
* `_set_grad()`：判定依据改成 `p.device`；
* `close()` **不再删** barrier 标记；残留由下次启动的 `ready` 握手统一清理；
* 握手改为**惰性** `_start()`，由 `broadcast()` / `all_reduce_grads()` 触发；
* 新增 `shard_indices(n, contiguous=True)` —— 数据并行必须各卡喂不同数据；
* 文档字符串补齐**使用契约**（params 顺序一致、冻结参数不要传、检查点只 rank0 写）
  与**等效 batch / 学习率**口径（默认平均 ⇒ 等效 `world × per_rank_batch`，lr 不自动改）。

### 验证

| 层 | 内容 | 结果 |
| --- | --- | --- |
| CPU 两进程 | 广播一致 + 梯度平均 [1,3]→2 | ✅ 连续 5 次全绿（修 ③ 前 1/3 概率挂） |
| CPU 两进程 | **某 rank 缺一层梯度**不卡死，按 0 补成 2.5 | ✅ 确定性复现（worker 里让 rank0 延迟 0.5 s） |
| 设备语义单测 | 把"numpy 对 None 返回 nan"这个坑本身钉住 | ✅ 7 条 |
| **同一张 V100 两个 rank** | 与单进程等效对拍：广播后 p=1.0/q=7.0；平均后 p_grad=2.0、q_grad=2.5；缺梯度侧 grad 也是 CUDA 张量 | ✅ `tools/dist_gpu_check.py` PASS |
| 本地套件 | — | **690 passed / 42 skipped** |
| V100 GPU 门 | 发版前跑 | **295 passed / 12 skipped / 0 failed** |

### 仍未验证（别当成已验）

* **真两张卡的加速比**：本机没有双卡机，加速比与主机中转开销都**没测**；
* **`dist_p2p.py`（CUDA IPC + `cuMemcpyPeerAsync`）完全未验证** —— 单卡上 peer copy 没意义；
* TP / PP 未实现。

现状与复现命令见 `docs/DISTRIBUTED_STATUS_2026-11-13.md`。
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
                d = z.read("tgai_go/train/dist.py").decode("utf-8", "replace")
                c = z.read("tgai_go/train/cuda.py").decode("utf-8", "replace")
                checks = {
                    "dist 有 copy_h2d_into 调用": "copy_h2d_into" in d,
                    "dist 有 shard_indices": "def shard_indices" in d,
                    "dist 惰性握手": "def _start" in d,
                    "dist 不删 barrier 标记": "不要删自己落下的 barrier" in d,
                    "cuda 有 copy_h2d_into": "def copy_h2d_into" in c,
                }
                for kk, hit in checks.items():
                    print("   %-24s %s" % (kk, hit))
                    ok = ok and hit
    print("产物:", [os.path.basename(f) for f in files])
    print("校验:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
