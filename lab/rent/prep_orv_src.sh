#!/bin/bash
# Prepare a scratch copy of the installed package at /root/orv_src/python/tgai_go
PY=/root/miniconda3/bin/python
PYLIB=$($PY -c 'import site;print(site.getsitepackages()[0])')
echo "  sitepackages = $PYLIB"
rm -rf /root/orv_src
mkdir -p /root/orv_src/python
cp -r "$PYLIB/tgai_go" /root/orv_src/python/tgai_go
ls -l /root/orv_src/python/tgai_go/train/cuda.py
$PY -c "import sys; sys.path.insert(0,'/root/orv_src/python'); import tgai_go; print('  version', tgai_go.__version__)"
