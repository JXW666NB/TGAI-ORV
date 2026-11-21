#!/bin/bash
# 找"DP OOM / ZeRO-2 跑得通"的 seq：seq768
unset PYTHONPATH
set -u
bash /root/x_zero4090.sh mem 768 2 3
bash /root/x_zero4090.sh mem 768 0 2
echo WRAPPER3_DONE
