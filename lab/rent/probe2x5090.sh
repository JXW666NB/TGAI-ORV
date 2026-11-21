#!/bin/bash
# Dual-5090 probe: hardware, P2P, NCCL, torch, and what the box can actually do
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
LOG=/root/probe2x.log
: > $LOG

echo "=== GPU ===" >> $LOG
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader >> $LOG 2>&1
nvidia-smi topo -m >> $LOG 2>&1

echo "=== torch ===" >> $LOG
$PY - >> $LOG 2>&1 <<'PY'
import torch
print(" torch", torch.__version__, "cuda", torch.version.cuda)
print(" devices", torch.cuda.device_count()),
print(" nccl available:", torch.distributed.is_nccl_available())
print(" nccl version:", torch.cuda.nccl.version() if hasattr(torch.cuda, "nccl") else "?")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print("  dev%d %s sm_%d%d  %.1f GiB" % (i, p.name, p.major, p.minor, p.total_memory/2**30))
if torch.cuda.device_count() >= 2:
    print(" p2p 0->1:", torch.cuda.can_device_access_peer(0, 1))
    print(" p2p 1->0:", torch.cuda.can_device_access_peer(1, 0))
PY

echo "=== CUDA Peer 直连测试（torch 层） ===" >> $LOG
$PY - >> $LOG 2>&1 <<'PY'
import torch
try:
    if torch.cuda.device_count() >= 2:
        a = torch.ones(1 << 20, device="cuda:0")
        b = torch.empty(1 << 20, device="cuda:1")
        b.copy_(a)
        torch.cuda.synchronize()
        print("  cross-device copy OK, sum =", float(b.sum()))
        # 计时：d2d 跨卡带宽
        import time
        n = 1 << 26                      # 256 MiB fp32
        x = torch.empty(n, device="cuda:0")
        y = torch.empty(n, device="cuda:1")
        for _ in range(3):
            y.copy_(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            y.copy_(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10
        print("  cross-device 256MiB copy: %.2f ms => %.1f GB/s"
              % (dt * 1e3, 2 * n * 4 / dt / 1e9))
except Exception as e:
    print("  failed:", type(e).__name__, str(e)[:200])
PY

echo "=== ORV dist 能力（装好后才有） ===" >> $LOG
$PY -m pip install -q -U --index-url https://pypi.org/simple --no-cache-dir \
    "tgai-orv==1.0.14" >> $LOG 2>&1
$PY -c "import tgai_go;print('  tgai_go',tgai_go.__version__)" >> $LOG 2>&1
$PY - >> $LOG 2>&1 <<'PY'
try:
    from tgai_go.train import dist
    print("  dist module:", dist.__doc__.strip().splitlines()[0] if dist.__doc__ else "(no doc)")
    names = [n for n in dir(dist) if not n.startswith("_")]
    print("  public:", names[:30])
except Exception as e:
    print("  dist import failed:", type(e).__name__, e)
PY
echo DONE >> $LOG
