#!/bin/bash
set -e
rm -rf /root/orv_src/python/tgai_go
mkdir -p /root/orv_src/python
tar xzf /root/bench/_pkg.tgz -C /root/orv_src/python
echo "== extracted =="
ls /root/orv_src/python/tgai_go/train/dist*.py
PYTHONPATH=/root/orv_src/python /root/miniconda3/bin/python -c "import tgai_go, tgai_go.train.dist_nccl as N; print(tgai_go.__version__, hasattr(N,'find_libnccl'))"
echo "== find nccl =="
PYTHONPATH=/root/orv_src/python /root/miniconda3/bin/python -c "from tgai_go.train import dist_nccl as N; l=N.find_libnccl(); print(l.path, l.version_str())"
echo "== backend env =="
PYTHONPATH=/root/orv_src/python TG_DP_BACKEND=nccl /root/miniconda3/bin/python -c "from tgai_go.train import dist; print(dist.dp_backend_requested())"
