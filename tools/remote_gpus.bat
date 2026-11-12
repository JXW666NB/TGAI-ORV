@echo off
powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"
