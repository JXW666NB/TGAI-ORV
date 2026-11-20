#!/bin/bash
# 5090 final comparison: ORV vs PyTorch (torch 2.8.0+cu128), both best settings
PY=/root/miniconda3/bin/python
export PYTHONIOENCODING=utf-8
cd /root/bench || exit 1
LOG=/root/f5090.log
: > $LOG
get() { grep -o "$1" | grep -o '[0-9.]*$'; }

echo "=== torch on this image ===" >> $LOG
$PY - >> $LOG 2>&1 <<'PY'
import torch
print("  torch", torch.__version__, "cuda", torch.version.cuda)
a = torch.ones(64, 64, device="cuda")
torch.cuda.synchronize()
print("  cuda matmul sum =", float((a @ a).sum()))
PY

echo "=== nvrtc ===" >> $LOG
$PY - >> $LOG 2>&1 <<'PY'
import ctypes, glob
for p in sorted(glob.glob("/usr/local/cuda*/lib64/libnvrtc.so.12*"))[:3]:
    try:
        lib = ctypes.CDLL(p)
        ma, mi = ctypes.c_int(), ctypes.c_int()
        lib.nvrtcVersion(ctypes.byref(ma), ctypes.byref(mi))
        print("  %-48s nvrtc %d.%d knows_120=%s"
              % (p, ma.value, mi.value, b"compute_120" in open(p, "rb").read()))
    except Exception as e:
        print("  %s failed %s" % (p, type(e).__name__))
PY

COMMON="--data /root/bench/data2048_v96000 --vocab 96000 --seq 512 --d 1024 \
--layers 12 --E 4 --n_q 16 --d_ff 3328 --top_k 2 --csa 4 --opt int8gpu \
--batch 1 --steps 20 --log-every 99 --no-final-save"

echo "=== ORV: amp off / amp gemm (2 rounds each) ===" >> $LOG
for amp in off gemm; do
  for r in 1 2; do
    out=$($PY -m tgai_go.train.v3_pretrain $COMMON --amp $amp 2>&1)
    v=$(echo "$out" | get '"step_s_steady": [0-9.]*')
    echo "  orv amp=$amp r=$r ${v:-FAIL}" >> $LOG
    if [ -z "$v" ]; then echo "$out" | tail -4 >> $LOG; fi
  done
done

echo "=== PyT: no amp / amp (2 rounds each) ===" >> $LOG
P="--d 1024 --layers 12 --E 4 --vocab 96000 --d_ff 3328 --n_heads 16 --top_k 2 \
--seq 512 --steps 20 --ckpt 0"
for r in 1 2; do
  v=$($PY v3_pyt_train.py $P 2>&1 | get '"step_s": [0-9.]*')
  echo "  pyt noamp r=$r ${v:-FAIL}" >> $LOG
done
for r in 1 2; do
  v=$($PY v3_pyt_train.py $P --amp 2>&1 | get '"step_s": [0-9.]*')
  echo "  pyt amp r=$r ${v:-FAIL}" >> $LOG
done
echo DONE >> $LOG
