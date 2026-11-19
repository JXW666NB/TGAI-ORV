# -*- coding: utf-8 -*-
"""V100 远控最小封装（argv 直传，绕开 PowerShell 引号抖动）。

用法（参数一律按空白分隔，不要带引号/空格）：
    python x.py run  dir /b D:\\orv_bench
    python x.py py   D:\\orv_bench\\foo.py --seq 2048
    python x.py pyt  D:\\orv_bench\\foo.py --seq 2048
    python x.py put  .\\a.py D:\\orv_bench\\a.py
    python x.py get  D:\\orv_bench\\out.json .\\out.json

约定：
  * `run` 远端用 cmd 执行原样命令；
  * `py`  远端 = ORV 解释器（D:\\orv_pip_0962\\Scripts\\python.exe），工作目录 D:\\orv_bench，
          且 PYTHONIOENCODING=utf-8；
  * `pyt` 同上，但解释器 = C:\\Python311\\python.exe（torch 2.6.0+cu124）。
输出直接写字节，避免本地 GBK 控制台把远端 UTF-8/中文打崩。
"""
import os
import subprocess
import sys

USER = "Administrator"
HOST = "192.168.0.103"
KEY = os.path.join(os.path.expanduser("~"), ".ssh", "v100_tgai")
ORV_PY = r"D:\orv_pip_0962\Scripts\python.exe"
PYT_PY = r"C:\Python311\python.exe"
WORK = r"D:\orv_bench"

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=20", "-i", KEY]


def _ssh(remote, timeout=None):
    cmd = ["ssh"] + SSH_OPTS + ["%s@%s" % (USER, HOST), remote]
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def _scp(args):
    cmd = ["scp"] + ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                     "-i", KEY] + list(args)
    return subprocess.run(cmd, capture_output=True)


def _emit(p):
    if p.stdout:
        sys.stdout.buffer.write(p.stdout)
        sys.stdout.buffer.flush()
    if p.stderr:
        sys.stdout.buffer.write(p.stderr)
        sys.stdout.buffer.flush()
    print("\n[rc=%d]" % p.returncode, flush=True)
    return p.returncode


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    act = sys.argv[1]
    rest = sys.argv[2:]
    if act == "run":
        return _emit(_ssh(" ".join(rest)))
    if act in ("py", "pyt"):
        # 远端默认 shell 是 PowerShell（实测 `dir /b` 被它抢去执行），
        # 所以不用 `set VAR=..&&` 那套 cmd 语法；改由 `-X utf8` 让解释器
        # 自己走 UTF-8 stdio，脚本内部自行 os.chdir(WORK)。
        exe = ORV_PY if act == "py" else PYT_PY
        remote = " ".join([exe, "-X", "utf8"] + rest)
        return _emit(_ssh(remote))
    if act == "put":
        local, remote = rest[0], rest[1]
        p = _scp([local, "%s@%s:%s" % (USER, HOST, remote)])
        return _emit(p)
    if act == "get":
        remote, local = rest[0], rest[1]
        p = _scp(["%s@%s:%s" % (USER, HOST, remote), local])
        return _emit(p)
    print("unknown action %r" % act)
    return 2


if __name__ == "__main__":
    sys.exit(main())
