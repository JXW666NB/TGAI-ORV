#!/bin/bash
# 鍦?4090 涓婂噯澶?琛ヤ竵鍖?锛氬鍒跺凡瀹夎鐨?tgai_go 鍒?/root/patched_pkg锛岄殢鍚庤鐩?3 涓敼鍔ㄦ枃浠?set -e
PYLIB=$(/root/miniconda3/bin/python -c "import site;print(site.getsitepackages()[0])")
echo "site-packages = $PYLIB"
rm -rf /root/patched_pkg
mkdir -p /root/patched_pkg
cp -r "$PYLIB/tgai_go" /root/patched_pkg/tgai_go
echo "宸插鍒跺埌 /root/patched_pkg/tgai_go"
ls /root/patched_pkg/tgai_go/train/ | head -5
