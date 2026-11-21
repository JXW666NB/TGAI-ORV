#!/bin/bash
# 核对：同一 seq2048 配置，装好的 1.0.14 wheel（不带 PYTHONPATH）vs 本地 HEAD 源码树
PY=/root/miniconda3/bin/python
cd /root/bench || exit 1
C="--data /root/bench/data2048_v96000 --vocab 96000 --seq 2048 --d 1024 --layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu --batch 1 --steps 20 --log-every 99 --no-final-save --amp off"
echo "== installed wheel 1.0.14 (site-packages) =="
env -u PYTHONPATH $PY -m tgai_go.train.v3_pretrain $C 2>&1 | grep -E "^step|^RESULT" | tail -3
echo "== local HEAD tree (/root/orv_src/python) =="
PYTHONPATH=/root/orv_src/python $PY -m tgai_go.train.v3_pretrain $C 2>&1 | grep -E "^step|^RESULT" | tail -3
