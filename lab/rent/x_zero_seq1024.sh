#!/bin/bash
# 继续找"DP OOM / ZeRO-2 跑得通"的 seq：先试 seq1024
unset PYTHONPATH
set -u
bash /root/x_zero4090.sh mem 1024 2 3
bash /root/x_zero4090.sh mem 1024 0 2
echo WRAPPER2_DONE
