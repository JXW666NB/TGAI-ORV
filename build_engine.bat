@echo off
REM ===========================================================================
REM  Build the CPU inference engine from THIS repo's src/ (Windows x64 / MSVC).
REM
REM  Why this file exists
REM    The old rebuild_cpu.bat hard-coded G=D:\tgai_nlp\TGAI GO, so it compiled
REM    the OLD GO tree instead of this repo's src/. Editing src/tgai_engine.c
REM    here had no effect, and on any other machine the script simply died.
REM    This one locates the repo with %~dp0 and finds the toolchain via vswhere.
REM
REM  Output: cpu\build\libtgai_engine.dll
REM  Usage : run from anywhere (double-click is fine).
REM ===========================================================================
setlocal enabledelayedexpansion
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

REM --- 1. locate vcvars64.bat (vswhere first, then common paths) -------------
set "VCVARS="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
    )
)
if not defined VCVARS (
    if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
if not defined VCVARS (
    if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
if not defined VCVARS (
    echo [ERROR] vcvars64.bat not found.
    echo         Fix: install Visual Studio Build Tools with the C++ toolset,
    echo         or pass another compiler to cpu\build.ps1.
    exit /b 1
)
echo [build_engine] VCVARS=%VCVARS%

call "%VCVARS%" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] vcvars64.bat failed.
    echo         Fix: run it alone to see the error; usually the Windows SDK.
    exit /b 1
)

REM --- 2. make sure we build this repo, not the old GO tree ------------------
if not exist "%ROOT%\src\tgai_engine.c" (
    echo [ERROR] %ROOT%\src\tgai_engine.c not found.
    echo         Fix: run this from the repository root.
    exit /b 1
)
echo [build_engine] SRC=%ROOT%\src

REM --- 3. call the official build script (it resolves ..\src relative to itself)
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\cpu\build.ps1" -Target windows
set "RC=%ERRORLEVEL%"
echo [build_engine] build.ps1 exit_code=%RC%

if exist "%ROOT%\cpu\build\libtgai_engine.dll" (
    echo [build_engine] OK cpu\build\libtgai_engine.dll
) else (
    echo [build_engine] no dll produced - see compiler output above
)
exit /b %RC%
