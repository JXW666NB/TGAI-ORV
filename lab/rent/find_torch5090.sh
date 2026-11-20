#!/bin/bash
# Find every python on the box and the torch each one has (the image ships torch 2.8 + cu128)
echo "=== pythons on PATH ==="
for p in /usr/bin/python3 /usr/local/bin/python3 /opt/conda/bin/python /root/miniconda3/bin/python; do
  if [ -x "$p" ]; then
    v=$($p - <<'PY'
import sys
try:
    import torch
    print("py%s torch=%s cuda=%s arch_ok=%s" % (
        sys.version.split()[0], torch.__version__, torch.version.cuda,
        "sm_120" in "".join(getattr(torch.cuda, "get_arch_list", lambda: [])())))
except Exception as e:
    print("py%s no-torch (%s)" % (sys.version.split()[0], type(e).__name__))
PY
)
    printf "  %-32s %s\n" "$p" "$v"
  fi
done

echo "=== which python3 resolves to ==="
command -v python3
python3 -c "import sys;print(' sys.executable =',sys.executable)"

echo "=== system python torch op test ==="
for p in /usr/bin/python3 /opt/conda/bin/python; do
  if [ -x "$p" ]; then
    echo "  --- $p ---"
    $p - <<'PY'
try:
    import torch
    print("   torch", torch.__version__, "cuda", torch.version.cuda)
    print("   arch_list:", torch.cuda.get_arch_list()[-6:])
    a = torch.ones(64, 64, device="cuda")
    torch.cuda.synchronize()
    print("   matmul sum =", float((a @ a).sum()))
except Exception as e:
    print("   failed:", type(e).__name__, str(e)[:160])
PY
  fi
done
