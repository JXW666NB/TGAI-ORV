@echo off
echo --- vulkan-1.dll ---
if exist C:\Windows\System32\vulkan-1.dll (echo YES) else (echo NO)
for %%d in (nvoglv64.dll nvoglv32.dll) do if exist C:\Windows\System32\%%d (echo %%d YES) else (echo %%d NO)
echo --- vulkaninfo ---
where vulkaninfo >nul 2>&1 && echo FOUND || echo NOT_FOUND
echo --- dlldir ---
dir /b C:\Windows\System32\vulkan*.dll 2>nul
