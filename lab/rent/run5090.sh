#!/bin/bash
# Fresh 5090 image: install ORV 1.0.12, gen data, then ORV vs PyTorch comparison
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
LOG=/root/run5090.log
: > $LOG
get() { grep -o "$1" | grep -o '[0-9.]*$'; }

echo "=== 1. env ===" >> $LOG
$PY -V >> $LOG 2>&1
$PY - >> $LOG 2>&1 <<'PY'
import torch
print("  torch", torch.__version__, "cuda", torch.version.cuda)
print("  arch_list tail:", torch.cuda.get_arch_list()[-5:])
a = torch.ones(64, 64, device="cuda")
torch.cuda.synchronize()
print("  cuda matmul sum =", float((a @ a).sum()))
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader >> $LOG 2>&1

echo "=== 2. install tgai-orv 1.0.12 ===" >> $LOG
$PY -m pip install -q -U --index-url https://pypi.org/simple --no-cache-dir \
    "tgai-orv==1.0.12" >> $LOG 2>&1
$PY -c "import tgai_go;print('  tgai_go',tgai_go.__version__)" >> $LOG 2>&1

echo "=== 3. gen data ===" >> $LOG
mkdir -p /root/bench && cd /root/bench || exit 1
[ -d /root/bench/data2048_v96000 ] || $PY /root/gen_data.py >> $LOG 2>&1
ls /root/bench/data2048_v96000 2>/dev/null | head -1 >> $LOG

COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--batch 1 --steps 20 --log-every 99 --no-final-save"

echo "=== 4. ORV (amp off / gemm, 2 rounds each) ===" >> $LOG
for amp in off gemm; do
  for r in 1 2; do
    out=$($PY -m tgai_go.train.v3_pretrain $COMMON --amp $amp 2>&1)
    v=$(echo "$out" | get '"step_s_steady": [0-9.]*')
    pk=$(echo "$out" | get '"pool_peak_live_gib": [0-9.]*')
    echo "  orv amp=$amp r=$r step=${v:-FAIL} peak=${pk:-?}" >> $LOG
    [ -z "$v" ] && echo "$out" | tail -4 >> $LOG
  done
done

echo "=== 5. PyT (noamp / amp, 2 rounds each) ===" >> $LOG
P="--d 1024 --layers 12 --E 4 --vocab 96000 --d_ff 3328 --n_heads 16 --top_k 2 \
--seq 512 --steps 20 --ckpt 0"
cp /root/v3_pyt_train.py /root/bench/v3_pyt_train.py 2>/dev/null
for r in 1 2; do
  v=$($PY v3_pyt_train.py $P 2>&1 | get '"step_s": [0-9.]*')
  echo "  pyt noamp r=$r ${v:-FAIL}" >> $LOG
done
for r in 1 2; do
  v=$($PY v3_pyt_train.py $P --amp 2>&1 | get '"step_s": [0-9.]*')
  echo "  pyt amp r=$r ${v:-FAIL}" >> $LOG
done
echo DONE >> $LOG
