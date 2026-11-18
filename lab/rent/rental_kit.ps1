# 租机一键测量脚本（RTX 4090 / 24GB 场景）
#
# 用法（在租来的机器上，PowerShell）：
#     powershell -NoProfile -ExecutionPolicy Bypass -File rental_kit.ps1
#
# 做什么：装 ORV → 打印设备信息 → 生成随机分片 → 小配置冒烟 → 1.5B/seq512 测步时。
# 全程只写自己的目录，不改系统设置；每步失败都会**停下来并打印原因**，
# 不会带着错误继续（免得烧钱跑出一堆无效数字）。
#
# 预算提示：4090 24GB 上 1.5B 只能跑 seq512（m8v16 约 20.2 GiB）；
#           seq1024 需 25.8 GiB，装不下。

$ErrorActionPreference = 'Stop'
$root = $env:TGAI_RENT_ROOT
if (-not $root) { $root = Join-Path $env:USERPROFILE 'orv_rent' }
New-Item -ItemType Directory -Force $root | Out-Null
Set-Location $root

function Step($name) { "`n===== $name =====" }

# ---------- 1. 设备信息 ----------
Step '1/5 设备信息'
try {
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
} catch {
    "找不到 nvidia-smi：这台机器可能不是 NVIDIA GPU 环境"
    exit 1
}
python -c "import sys; print('python', sys.version.split()[0])"

# ---------- 2. 安装 ORV（指定版本，避免装到别的）----------
Step '2/5 安装 tgai-orv==0.9.97'
python -m pip install -q --upgrade pip 2>$null
python -m pip install -q tgai-orv==0.9.97
if ($LASTEXITCODE -ne 0) { "安装失败：网络或 pip 源问题"; exit 1 }
python -c "import tgai_go; print('tgai-orv', tgai_go.__version__)"

# ---------- 3. 生成随机分片 ----------
Step '3/5 生成 packed 分片（随机 token，仅用于测性能）'
$data = Join-Path $root 'data512'
if (-not (Test-Path $data)) {
    python make_packed.py --out $data --seq 512 --vocab 96000 --shards 16 --rows 64
    if ($LASTEXITCODE -ne 0) { "数据生成失败"; exit 1 }
} else { "已存在：$data" }

# ---------- 4. 小配置冒烟（确认内核能在本卡上编译）----------
Step '4/5 冒烟：2 层 / seq512 / 3 步'
python -m tgai_go.train.v3_pretrain --data $data --vocab 96000 --seq 512 `
    --d 1024 --layers 2 --E 4 --n_q 16 --d_ff 2816 --top_k 2 --csa 4 `
    --steps 3 --log-every 1 --no-final-save
if ($LASTEXITCODE -ne 0) {
    "冒烟失败 —— 可能是 sm_89（4090）上某个内核编译不过。把上面的报错贴回去。"
    exit 1
}

# ---------- 5. 1.5B / seq512 测步时 ----------
Step '5/5 1.5B / seq512 / 6 步（m8v16 优化器，约 20.2 GiB）'
python -m tgai_go.train.v3_pretrain --data $data --vocab 96000 --seq 512 `
    --d 1024 --layers 24 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 `
    --opt fp16mom --opt-state bf16 --steps 6 --log-every 1 --no-final-save
if ($LASTEXITCODE -ne 0) {
    "1.5B 失败 —— 若报显存不足，说明这张卡的实际可用显存低于 20.2 GiB。"
    exit 1
}

Step '完成'
"对照基准（V100 32GB）：1.5B/seq512 = 0.446 s/步，987 tok/s"
"把上面的 loss/tok/s 那几行贴回来即可分析。"
