# -*- coding: utf-8 -*-
"""远程执行小工具（参数列表直传，避免 PowerShell 抢引号/抢 `&`）。

用法：
    python lab/rent/rsx.py put  <本地文件> <远端路径>
    python lab/rent/rsx.py get  <远端路径> <本地文件>
    python lab/rent/rsx.py run  <远端命令行>          # 单条命令
    python lab/rent/rsx.py py   <本地脚本> [额外参数...]  # put 到 D:\\orv_bench 后用 ORV python 跑
    python lab/rent/rsx.py pyt  <本地脚本> [额外参数...]  # put 到 D:\\orv_bench 后用 torch python 跑

为什么存在：`ssh ... "cmd & cmd"` 里 `&` 会被本地 PowerShell 当运算符；
`v100_ssh.py run` 只做 `" ".join(argv)`，`&`/引号仍要经 PowerShell 一层。
这里所有远端命令都先落到本地临时 `.ps1`/字符串再由 subprocess argv 直传，
远端命令行里**不出现** `&`、`|`、`>`、`;` 这类会被 cmd 抢的字符。
"""
import argparse
import os
import subprocess
import sys

USER = "Administrator"
HOST = "192.168.0.103"
KEY = os.path.join(os.path.expanduser("~"), ".ssh", "v100_tgai")
ORV_PY = r"D:\orv_pip_0962\Scripts\python.exe"
PYT_PY = r"C:\Python311\python.exe"
REMOTE_BENCH = r"D:\orv_bench"

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=20", "-i", KEY]


def _emit(p):
    if p.stdout:
        sys.stdout.buffer.write(p.stdout)
        sys.stdout.buffer.flush()
    if p.stderr:
        sys.stderr.buffer.write(p.stderr)
        sys.stderr.buffer.flush()
    return p.returncode


def ssh(cmdline):
    return _emit(subprocess.run(["ssh"] + SSH_OPTS + ["%s@%s" % (USER, HOST), cmdline],
                                capture_output=True))


def scp(local, remote):
    return _emit(subprocess.run(["scp", "-o", "BatchMode=yes",
                                 "-o", "StrictHostKeyChecking=no", "-i", KEY,
                                 local, "%s@%s:%s" % (USER, HOST, remote)],
                                capture_output=True))


def scp_get(remote, local):
    return _emit(subprocess.run(["scp", "-o", "BatchMode=yes",
                                 "-o", "StrictHostKeyChecking=no", "-i", KEY,
                                 "%s@%s:%s" % (USER, HOST, remote), local],
                                capture_output=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "put", "get", "py", "pyt"])
    ap.add_argument("args", nargs="+")
    a = ap.parse_args()
    if a.action == "run":
        return ssh(" ".join(a.args))
    if a.action == "put":
        return scp(a.args[0], a.args[1])
    if a.action == "get":
        return scp_get(a.args[0], a.args[1])
    local = a.args[0]
    extra = a.args[1:]
    remote = REMOTE_BENCH + "\\" + os.path.basename(local)
    rc = scp(local, remote)
    if rc != 0:
        print("[rsx] scp failed rc=%d" % rc)
        return rc
    py = ORV_PY if a.action == "py" else PYT_PY
    # 远端：set PYTHONIOENCODING 后跑脚本（用 `&&` 不行，cmd 下用 `set X=1 & ...` 会被抢，
    # 所以写一个临时 bat 再执行 —— 只出现一次重定向，由 cmd 自己处理）。
    bat_local = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "_rsx_run.bat")
    with open(bat_local, "w", newline="\r\n") as f:
        f.write("@echo off\r\n")
        f.write("set PYTHONIOENCODING=utf-8\r\n")
        f.write("cd /d %s\r\n" % REMOTE_BENCH)
        f.write('"%s" "%s" %s\r\n' % (py, remote, " ".join('"%s"' % x for x in extra)))
    rc = scp(bat_local, REMOTE_BENCH + "\\_rsx_run.bat")
    if rc != 0:
        print("[rsx] bat scp failed rc=%d" % rc)
        return rc
    return ssh("%s\\_rsx_run.bat" % REMOTE_BENCH)


if __name__ == "__main__":
    sys.exit(main())
