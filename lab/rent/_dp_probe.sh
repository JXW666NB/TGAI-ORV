#!/bin/bash
PY=/root/miniconda3/bin/python
S=/root/miniconda3/lib/python3.12/site-packages/tgai_go/train
echo "== version =="
$PY -c "import tgai_go; print(tgai_go.__version__)"
echo "== installed v3_pretrain DataParallel hits =="
grep -c "DataParallel" $S/v3_pretrain.py || true
echo "== installed dist TG_DP / host_register =="
grep -c "TG_DP_BACKEND" $S/dist.py || true
grep -c "host_register" $S/dist.py || true
echo "== torch =="
$PY -c "import torch; print(torch.__version__, torch.version.cuda)"
echo "== nccl libs =="
ls -l /root/miniconda3/lib/python3.12/site-packages/nvidia/*/lib/libnccl.so* 2>/dev/null
ls -l /root/miniconda3/lib/python3.12/site-packages/torch/lib/libnccl* 2>/dev/null
echo "== ldconfig =="
ldconfig -p 2>/dev/null | grep -i nccl || echo "(none)"
echo "== bench =="
ls -la /root/bench
echo "== v3_pyt_train head =="
head -40 /root/bench/v3_pyt_train.py
echo "== v3_pyt_train ddp =="
grep -n "DistributedDataParallel\|init_process_group\|torchrun\|local_rank" /root/bench/v3_pyt_train.py | head -20
