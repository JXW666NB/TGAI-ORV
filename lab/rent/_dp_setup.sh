#!/bin/bash
set -e
S=/root/miniconda3/lib/python3.12/site-packages/tgai_go
rm -rf /root/orv_src/python/tgai_go
mkdir -p /root/orv_src/python
cp -a "$S" /root/orv_src/python/tgai_go
find /root/orv_src/python -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
echo "== copied =="
ls /root/orv_src/python/tgai_go/train | wc -l
echo "== md5 of installed originals =="
md5sum "$S/train/dist.py" "$S/train/v3_pretrain.py" "$S/train/cuda.py" "$S/__init__.py"
echo "== df =="
df -h /root | tail -1
