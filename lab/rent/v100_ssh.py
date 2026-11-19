# -*- coding: utf-8 -*-
"""V100 租机（Windows）远控助手：用 subprocess 直接给 ssh/scp 传**参数列表**。

为什么不用 `ssh ... "命令"` 手写：PowerShell 那层会在引号上抖动（实测 `&&`/`dir /b`
被本地 PowerShell 抢去执行）。这里所有引号都由 subprocess 的 argv 规则处理，
远端只收到一条完整命令行。

用法：
    python lab/rent/v100_ssh.py run  "dir /b D:\\orv_bench"
    python lab/rent/v100_ssh.py run  "D:\\orv_pip_0962\\Scripts\\python.exe D:\\orv_bench\\x.py" --timeout 3600
    python lab/rent/v100_ssh.py put  local.py "D:\\orv_bench\\x.py"
    python lab/rent/v100_ssh.py get  "D:\\orv_bench\\out.json" .\\out.json
"""
import argparse
import os
import subprocess
import sys

USER = "Administrator"
HOST = "192.168.0.103"
KEY = os.path.join(os.path.expanduser("~"), ".ssh", "v100_tgai")

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=20", "-i", KEY]


def _ssh(argv):
    cmd = ["ssh"] + SSH_OPTS + ["%s@%s" % (USER, HOST)] + list(argv)
    return subprocess.run(cmd, capture_output=True)


def _scp(argv):
    cmd = ["scp"] + ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                     "-i", KEY] + list(argv)
    return subprocess.run(cmd, capture_output=True)


def _emit(p):
    # 直接写字节：本地控制台是 GBK，远端可能吐 UTF-8 / 非法字节，
    # 走文本层会 UnicodeEncodeError（实测踩过）。
    if p.stdout:
        sys.stdout.buffer.write(p.stdout)
        sys.stdout.buffer.flush()
    if p.stderr:
        sys.stderr.buffer.write(p.stderr)
        sys.stderr.buffer.flush()
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "put", "get"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--timeout", type=float, default=1800.0)
    a = ap.parse_args()

    if a.action == "run":
        remote = " ".join(a.args)
        try:
            p = _ssh([remote])
        except subprocess.TimeoutExpired:
            print("[timeout] %s" % remote)
            return 124
        return _emit(p)

    if a.action == "put":
        local, remote = a.args[0], a.args[1]
        p = _scp([local, "%s@%s:%s" % (USER, HOST, remote)])
        rc = _emit(p)
        print("[put] %s -> %s rc=%d" % (local, remote, rc))
        return rc

    remote, local = a.args[0], a.args[1]
    p = _scp(["%s@%s:%s" % (USER, HOST, remote), local])
    rc = _emit(p)
    print("[get] %s -> %s rc=%d" % (remote, local, rc))
    return rc


if __name__ == "__main__":
    sys.exit(main())
