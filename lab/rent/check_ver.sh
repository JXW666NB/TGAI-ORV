#!/bin/bash
# 鏍稿锛歱atched_pkg 涓?repo HEAD 鐨?5 涓枃浠跺搱甯岋紱涓よ竟鐨?__version__
echo "=== /root/patched_pkg/tgai_go/train 鐨勫搱甯岋紙鍓?16 浣嶏級==="
cd /root/patched_pkg/tgai_go/train || exit 1
for f in cuda.py nn_ops.py v3_pretrain.py v3_train_cli.py __init__.py; do
  printf "  %-18s %s\n" "$f" "$(sha256sum "$f" | cut -c1-16)"
done

echo "=== 鐗堟湰 ==="
/root/miniconda3/bin/python - <<'PY'
import importlib, os, subprocess, sys
def ver(ppath=None):
    env = dict(os.environ)
    if ppath:
        env["PYTHONPATH"] = ppath
    else:
        env.pop("PYTHONPATH", None)
    out = subprocess.run([sys.executable, "-c",
                          "import tgai_go;print(getattr(tgai_go,'__version__','?'))"],
                         capture_output=True, text=True, env=env)
    return (out.stdout or out.stderr).strip()
print("  site-packages :", ver(None))
print("  patched_pkg   :", ver("/root/patched_pkg"))
PY

echo "=== pip 璁板綍 ==="
/root/miniconda3/bin/python -m pip show tgai-orv 2>/dev/null | head -3
