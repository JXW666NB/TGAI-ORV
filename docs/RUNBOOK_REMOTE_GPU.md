# 外出运维手册：远程 GPU 上跑 ORV（含外部人员离线模式）

> 2026-11-21 版　｜　适用：租机（AutoDL/seetacloud 等 SSH）、朋友/第三方机器（离线 zip）
>
> **当前状况**：V100 不在手边。现成的实验台只有两种 —— ①自己租的云卡（SSH）；
> ②朋友的 P100（**离线 zip 模式**，不发凭据、不给 SSH）。
> 这份手册的目标：**任何人在任何一台 GPU 机器上，照着做就能产出可对比、可复现的数据。**

---

## 0. 四条红线（写在最前面，违反任一条就停）

1. **凭据绝不进仓库**。密码/token 只放两处：`%TEMP%\orv_rent.json`（租机）或环境变量。
   `git grep` 能搜到的凭据等于已泄露；仓库是公开的。
2. **不碰别人机器上的运行中任务**。只跑独立的 benchmark 进程；不 kill、不改配置、不装驱动。
3. **算力按分钟计费**。单次实验设计 ≤5 分钟 GPU 时间；能先本地算的先本地算（见 §4 省钱清单）。
4. **只读优先**。排查他人机器时先 `nvidia-smi` / `nvidia-smi -q` / `df -h` / `python -V`，
   确认环境后再动手；写操作（装包、写文件）前先说清楚要写什么。

---

## 1. 两种通道，先选对

| 场景 | 通道 | 工具 | 凭据 |
| --- | --- | --- | --- |
| 自己租的云卡（有 SSH） | **在线 SSH** | `lab/rent/rsh.py`（推荐）/ `lab/rent/rem.py` | `%TEMP%\orv_rent.json` |
| 朋友的机器 / 客户内网 / 无 SSH | **离线 zip** | 打一个 zip 让对方跑，回传结果 | 不需要 |
| 有一个能开的网页 CDP（如 Electron/Chrome 远程调试） | 可选：被动观察 | `mcp__rea__*` 系列 | 不需要密码 |

### 1.1 在线 SSH（租机）

```powershell
# 1) 凭据文件（不在仓库里，别提交）
#    %TEMP%\orv_rent.json
#    {"host":"connect.westb.seetacloud.com","port":38451,"user":"root","password":"******"}

# 2) 连通性 + 环境体检（一次性，30 秒）
python lab/rent/rsh.py --cmd "nvidia-smi; nvidia-smi -L; python3 -V; df -h /root | tail -1"

# 3) 传脚本
python lab/rent/rsh.py --put bench/p100_fair/bench_fair.py /root/bench_fair.py

# 4) 跑（给足 timeout，长任务别用默认值）
python lab/rent/rsh.py --cmd "cd /root && python3 bench_fair.py --probe" --timeout 300
python lab/rent/rsh.py --cmd "cd /root && python3 bench_fair.py --bs 2,4 --iters 5 --json fair.json" --timeout 3600

# 5) 取回结果
python lab/rent/rsh.py --get /root/fair.json ./lab/results/20261121_v100/fair.json
```

> `rem.py` 是同一件事的早期版本（凭据走 `REM_HOST/REM_PORT/REM_USER/REM_PASS` 环境变量）。
> 两者都只依赖 `paramiko`；`rsh.py` 更好用（自动 utf-8、支持 put/get）。

### 1.2 离线 zip（朋友 / 第三方）——**当前 P100 就是这个模式**

**不要**索取对方 SSH；**不要**让对方装驱动。只做三件事：

1. 你这边：`Compress-Archive bench\p100_fair\* -DestinationPath orv_fair.zip`
2. 对方：解压 → `pip install -U tgai-orv==<版本>` → 按 `优化说明_发现的问题.md` 里的顺序跑
3. 对方回传：`fair.json`、`ladder.json`、`results_template.md`（填过）+ 关键 `pool_dump_live` 输出

**给对方的最小操作说明**（可直接复制进聊天）：

```text
1) pip install -U tgai-orv==1.0.20
2) python bench_fair.py --probe                # 先贴这段输出
3) python bench_fair.py --bs 2,4 --iters 5 --json fair.json
4) python bench_ladder.py --bs 4 --iters 4 --json ladder.json
5) 把 fair.json / ladder.json / results_template.md 回传
注意：torch 与 ORV 必须分进程跑；跑之前别开着别的 GPU 任务。
```

---

## 2. 出门前的 5 分钟检查（本地做完，省 GPU 钱）

```powershell
# ① 语法与自检（不花 GPU）
python -m py_compile bench\p100_fair\bench_fair.py
python -X utf8 -m pytest tests -q --no-header            # 本地全绿再出门

# ② 打包前确认脚本里没有写死本机路径
Select-String -Path bench\p100_fair\*.py -Pattern "D:\\|C:\\Users"

# ③ 版本号与 CHANGELOG 一致（远端会 pip install 你发的版本）
python -c "import sys; sys.path.insert(0,'python'); import tgai_go; print(tgai_go.__version__)"
```

---

## 3. 到了机器上：固定顺序（照抄即可）

```bash
# ---- A. 环境体检（30 秒，先看清再动手）----
nvidia-smi                     # 型号 / 显存 / 驱动 / 是否有别的进程占卡
nvidia-smi -L
python3 -V
df -h /root | tail -1
echo "CC=$(python3 -c 'import torch;print(torch.cuda.get_device_capability(0))' 2>/dev/null || echo no-torch)"

# ---- B. 装 ORV（不装 torch 也能跑 ORV 侧）----
pip install -U tgai-orv==<版本>

# ---- C. 冒烟（10 秒，失败就别往下跑）----
python3 -c "import tgai_go;print(tgai_go.__version__, tgai_go.native_available())"
python3 -c "import tgai_go.train as T;print(T.policy())"      # 看策略是否 auto/flash

# ---- D. 探 SDPA 后端（决定"物化 vs 不物化"能不能测）
python3 bench_fair.py --probe

# ---- E. 正式测（分进程！）----
python3 bench_fair.py --bs 2,4 --iters 5 --json fair.json
python3 bench_ladder.py --bs 4 --iters 4 --json ladder.json

# ---- F. 显存归因（最关键的一份数据）----
TG_POOL_DEBUG=1 python3 -c "
import tgai_go.train as T
rt = T._cuda()
print(rt.pool_stats())
rt.pool_dump_live(top=20)
"
```

**判断 cc（算力等级）**：`cc < 7.0` ⇒ Pascal（P100 等），**没有 Tensor Core**，
ORV 的加速路径不成立，只测显存与稳定性，不要期待速度优势；`cc >= 7.0`（V100/T4）才谈加速，
`cc >= 8.0`（A100/4090/5090）才是 ORV 的主场。

---

## 4. 省钱清单（每次都省）

| 做法 | 省多少 |
| --- | --- |
| 本地能算的绝不搬 GPU（显存估算、参数计数、语法自检） | 每次 2–5 分钟 |
| 一次 SSH 传完所有脚本、一次跑完所有 batch | 每次 3–10 分钟 |
| `--iters 4` 起步，有异常再加大 | 30–50% GPU 时间 |
| 先 `--probe`/`--bs 2` 冒烟，不直接上 B=8 | 避免 OOM 白跑 |
| 跑完立刻 `shutdown`（租机按小时计费） | 直接省一整段 |
| 失败先看日志再重跑，不盲试 | 避免重复烧钱 |

---

## 5. 结果归档（必须做，否则数据白拿）

目录约定：

```
lab/results/<日期>_<机器>/         例如 lab/results/20261121_p100/
    env.txt             # nvidia-smi、python/torch/orv 版本、cc
    fair.json           # bench_fair.py 产出
    ladder.json         # bench_ladder.py 产出
    pool_dump.txt       # pool_dump_live 输出
    notes.md            # 一句话结论 + 异常
```

`lab/results/` 进版本库（数据小、可复现），**大文件（>5 MB）不进**。

---

## 6. 故障排查表

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| `SSH banner error` / 连不上 | 实例已关机 / 到期 / 公网 IP 变 | 去控制台确认实例状态，重新取端口 |
| Windows 控制台中文乱码 | GBK vs UTF-8 | `rsh.py` 已强制 utf-8；本地用 `python -X utf8` |
| `Torch not compiled with CUDA` | 装的是 CPU 版 torch | 只跑 ORV 侧，或装 CUDA 版 |
| 同一进程先 torch 后 ORV 报"显存不足" | torch caching allocator 占住 | **分进程跑**（对方已发现，正确） |
| `attn_bwd_flash_w maxdyn 失败` | 0.9.x 在 P100 上的旧 bug | 用 1.0.18+ |
| `CUDA_ERROR_NO_BINARY_FOR_GPU` | 架构不匹配 | 用 `TGAI_CUDA_BIN` 指目标机自编译，或换预编译缓存 |
| NVRTC 报找不到编译器 | 无 nvcc/驱动不全 | 安装 CUDA toolkit 或设 `TGAI_CC` |
| 池峰值远高于 nvidia-smi | 池缓存没归还 | 报给引擎侧（`pool_stats().cached_bytes`），可 `TG_POOL_GB` 限池 |

---

## 7. V100 回来之后的复测清单（一页备查）

> V100 = cc 7.0，**有 fp16 Tensor Core、无 bf16 张量核**，是"ORV 应该赢"的第一档卡。

```bash
python3 bench_fair.py --probe                       # 确认 SDPA 后端
python3 bench_fair.py --bs 2,4,8 --iters 5 --json fair.json
python3 bench_ladder.py --bs 4,8 --iters 4 --json ladder.json
# 重点看三组：
#   A: 两边 fp32 + 低显存注意力   → 显存差距还有多少
#   C: 两边 fp16                  → 速度比（V100 上 fp16 才是公平档）
#   ladder 里 fp16=True 的最优行   → ORV 在 V100 上的最好配置
```

验收判据：**C 组速度 R ≤ 1.0 且显存差距 ≤10%** ⇒ 可以在 V100 上宣称与 PyTorch 平齐/更好；
达不到就继续按 `pool_dump_live` 定位显存，不要先改宣传口径。

---

## 8. 已知仍未解决的三件事（别在手册里假装解决了）

1. **P100 显存高 26~37%**：原因未定位（候选：MoE dispatch 缓冲 / 激活保留 / 池高水位），
   需要 §3-F 的 `pool_dump_live` 输出。
2. **`T.amp()` 不省显存**（设计如此：fp32 master + fp16 影子缓存），在 P100 上还会更慢；
   显存只能靠激活存 bf16 / 梯度重算。
3. **多卡重叠归约（`TG_DP_OVERLAP`）** NCCL 路径 gnorm 差 10.3%，等价性未证明 ⇒ 默认关闭。

---

## 9. 附：本文档涉及的文件

| 文件 | 作用 |
| --- | --- |
| `lab/rent/rsh.py` | 租机 SSH：`--cmd` / `--put` / `--get`，凭据读 `%TEMP%\orv_rent.json` |
| `lab/rent/rem.py` | 早期版本，凭据走 `REM_*` 环境变量 |
| `bench/p100_fair/bench_fair.py` | 公平 2×2 对照（精度 × 注意力路径，三口径显存）|
| `bench/p100_fair/bench_ladder.py` | ORV 旋钮阶梯（找该卡最优配置）|
| `bench/p100_fair/优化说明_发现的问题.md` | 给外部人员的完整发现清单 |
| `docs/P100_MEASURED_2026-11-21.md` | P100 实测记录（含不利结论）|
| `docs/P100_FAIR_RETEST.md` | 复测口径与判据 |
