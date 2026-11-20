#!/bin/bash
# 鐢?*鍙戝竷鐨?1.0.11 杞瓙**閲嶈锛屽啀鐢ㄨ濂界殑鍖咃紙涓嶈 PYTHONPATH锛夊璺戜笁涓暟瀛?cd /root/bench || exit 1
export PYTHONIOENCODING=utf-8
PY=/root/miniconda3/bin/python
LOG=/root/verify_1011.log
get() { grep -o "$1" | grep -o '[0-9.]*$'; }
: > $LOG

echo "=== 瀹夎 1.0.11 杞瓙锛?-force-reinstall锛岀粫寮€婊炲悗闀滃儚锛?==" >> $LOG
$PY -m pip install -q --force-reinstall --no-deps --no-cache-dir \
    /root/tgai_orv-1.0.11-py3-none-any.whl >> $LOG 2>&1
$PY -c "import tgai_go; print('  installed __version__ =', tgai_go.__version__)" >> $LOG 2>&1
$PY -c "import tgai_go.train as T; print('  _muon_default_ns_fast() =', T._muon_default_ns_fast())" >> $LOG 2>&1

B="--data /root/bench/data2048_v96000 --vocab 96000 --d 1024 --layers 12 --E 4 \
--n_q 16 --d_ff 3328 --top_k 2 --csa 4 --amp off --batch 1 --log-every 99 --no-final-save"

echo "--- seq512 / int8gpu / 20 姝?---" >> $LOG
v=$($PY -m tgai_go.train.v3_pretrain $B --seq 512 --opt int8gpu --steps 20 --no-final-save 2>&1 | get '"step_s_steady": [0-9.]*')
echo "  seq512_int8gpu = $v" >> $LOG

echo "--- seq512 / muon锛堟柊榛樿锛?/ 4 姝?---" >> $LOG
v=$($PY -m tgai_go.train.v3_pretrain $B --seq 512 --opt muon --steps 4 --no-final-save 2>&1 | get '"step_s_steady": [0-9.]*')
echo "  seq512_muon_fast = $v" >> $LOG

echo "--- seq512 / muon + parity / 4 姝?---" >> $LOG
v=$(TG_MUON_NS_PARITY=1 $PY -m tgai_go.train.v3_pretrain $B --seq 512 --opt muon --steps 4 --no-final-save 2>&1 | get '"step_s_steady": [0-9.]*')
echo "  seq512_muon_parity = $v" >> $LOG

echo "--- seq2048 + ckpt / int8gpu / 10 姝?---" >> $LOG
v=$(TG_GRAD_CKPT=1 $PY -m tgai_go.train.v3_pretrain $B --seq 2048 --opt int8gpu --steps 10 --no-final-save 2>&1 | get '"step_s_steady": [0-9.]*')
echo "  seq2048_ckpt_int8gpu = $v" >> $LOG

echo DONE >> $LOG
