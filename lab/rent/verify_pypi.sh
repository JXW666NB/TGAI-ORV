#!/bin/bash
# 浠?PyPI 骞插噣瀹夎 1.0.11锛岀劧鍚庣敤**瑁呭ソ鐨勫寘**锛堜笉璁?PYTHONPATH锛夊璺戜笁涓ご鏉℃暟瀛?cd /root/bench || exit 1
export PYTHONIOENCODING=utf-8
PY=/root/miniconda3/bin/python
LOG=/root/verify_pypi.log
get() { grep -o "$1" | grep -o '[0-9.]*$'; }
: > $LOG

echo "=== pip install -U tgai-orv==1.0.11 ===" >> $LOG
$PY -m pip install -q -U "tgai-orv==1.0.11" >> $LOG 2>&1
$PY -c "import tgai_go; print('installed __version__ =', tgai_go.__version__)" >> $LOG 2>&1

B="--data /root/bench/data2048_v96000 --vocab 96000 --d 1024 --layers 12 --E 4 \
--n_q 16 --d_ff 3328 --top_k 2 --csa 4 --amp off --batch 1 --log-every 99 --no-final-save"

echo "--- seq512 / adamw(int8gpu) / 20 姝?---" >> $LOG
v=$($PY -m tgai_go.train.v3_pretrain $B --seq 512 --opt int8gpu --steps 20 --no-final-save 2>&1 | get '"step_s_steady": [0-9.]*')
echo "  seq512_int8gpu step=$v" >> $LOG

echo "--- seq512 / muon 蹇矾寰?/ 4 姝?---" >> $LOG
v=$($PY -m tgai_go.train.v3_pretrain $B --seq 512 --opt muon --steps 4 --no-final-save 2>&1 | get '"step_s_steady": [0-9.]*')
echo "  seq512_muon_fast step=$v" >> $LOG

echo "--- seq512 / muon parity / 4 姝?---" >> $LOG
v=$(TG_MUON_NS_PARITY=1 $PY -m tgai_go.train.v3_pretrain $B --seq 512 --opt muon --steps 4 --no-final-save 2>&1 | get '"step_s_steady": [0-9.]*')
echo "  seq512_muon_parity step=$v" >> $LOG

echo "--- seq2048 + ckpt / int8gpu / 10 姝?---" >> $LOG
v=$(TG_GRAD_CKPT=1 $PY -m tgai_go.train.v3_pretrain $B --seq 2048 --opt int8gpu --steps 10 --no-final-save 2>&1 | get '"step_s_steady": [0-9.]*')
echo "  seq2048_ckpt_int8gpu step=$v" >> $LOG

echo DONE >> $LOG
