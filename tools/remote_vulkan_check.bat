@echo off
echo === vulkan-1.dll ===
if exist C:\Windows\System32\vulkan-1.dll (echo YES) else (echo NO)
echo === vulkaninfo ===
where vulkaninfo 2>nul || echo NOT_IN_PATH
echo === 各显卡驱动日期/版本 ===
powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | ForEach-Object { '{0} | {1} | {2}' -f $_.Name, $_.DriverVersion, $_.DriverDate }"
