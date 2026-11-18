#!/usr/bin/env bash
# Vulkan 设备可见性诊断 + torch 状态
echo "--- ICD 内容 ---"
cat /etc/vulkan/icd.d/nvidia_icd.json 2>&1
echo
echo "--- libGLX_nvidia 是否导出 ICD 入口（vk_icdGetInstanceProcAddr）---"
nm -D /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 2>/dev/null | grep -c icd
echo "--- 用显式 ICD 再枚举一次 ---"
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  vulkaninfo --summary 2>&1 | grep -iE 'deviceName|deviceType|GPU id|apiVersion|ERROR' | head -8
echo "--- 设备节点 ---"
ls /dev/nvidia* 2>&1 | head -6
echo "--- torch ---"
/root/miniconda3/bin/python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.device_count())' 2>&1 | tail -2
tail -3 /root/orv_rent/torch_install.log 2>&1
