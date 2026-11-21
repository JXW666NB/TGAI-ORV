#!/bin/bash
# Vulkan capability probe on the 5090: ICD present? device list? device-group count?
echo "=== libvulkan / ICD ==="
ldconfig -p 2>/dev/null | grep -i vulkan | head -5
ls -1 /usr/share/vulkan/icd.d/ 2>/dev/null | head -5 || echo "  no icd.d"
echo "=== vulkaninfo? ==="
command -v vulkaninfo >/dev/null && echo "  present" || echo "  absent (will try to install)"
if ! command -v vulkaninfo >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq vulkan-tools) >/dev/null 2>&1
  command -v vulkaninfo >/dev/null && echo "  installed now" || echo "  install failed"
fi

echo "=== vulkaninfo summary ==="
if command -v vulkaninfo >/dev/null; then
  vulkaninfo --summary 2>&1 | head -40
else
  echo "  skipping"
fi

echo "=== device group / external memory extensions ==="
if command -v vulkaninfo >/dev/null; then
  vulkaninfo 2>/dev/null | grep -iE "deviceGroup|device_group|external_memory|external_semaphore" | head -10
  echo "--- group counts ---"
  vulkaninfo 2>/dev/null | grep -iA3 "deviceGroupProperties\|physicalDeviceGroup" | head -12
fi

echo "=== CUDA side (for comparison on this single card) ==="
/root/miniconda3/bin/python - <<'PY'
import torch
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print("  cuda devices:", n, [torch.cuda.get_device_name(i) for i in range(n)])
    print("  p2p 0->1:", torch.cuda.can_device_access_peer(0, 1) if n > 1 else "n/a")
PY
