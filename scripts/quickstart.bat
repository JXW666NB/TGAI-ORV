@echo off
REM TGAI GO 训练引擎 · 一条命令自检（Windows）
setlocal
cd /d "%~dp0.."
set HERE=%CD%
echo 仓库：%HERE%
where python >nul 2>nul || (echo 找不到 python，请安装 Python 3.8+ 并勾选 Add to PATH & exit /b 1)
python -V
python -c "import numpy" 2>nul || (echo -- pip install numpy & python -m pip install -q numpy)
where nvidia-smi >nul 2>nul && (nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader) || echo 警告：没检测到 nvidia-smi
set PYTHONPATH=%HERE%\python;%PYTHONPATH%
echo PYTHONPATH=%PYTHONPATH%
echo ===== 自检（12 项）=====
python -m tgai_go.train.doctor
if errorlevel 1 (echo 自检未通过：按上面每项的提示处理 & exit /b 1)
echo.
echo 自检通过。下一步：
echo   训练：set PYTHONPATH=%HERE%\python ^&^& python examples\exam2_chat10m_tgai.py --device cuda --steps 600 --out chat10m
echo   验收：set PYTHONPATH=%HERE%\python ^&^& python tools\accept.py --quick
