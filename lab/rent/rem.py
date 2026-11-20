# -*- coding: utf-8 -*-
"""极简远程执行器（paramiko），供 4090 实例使用。

用法：
    python rem.py run  "nvidia-smi"
    python rem.py put  <本地文件> <远端路径>
    python rem.py get  <远端文件> <本地路径>

凭据从环境变量读：REM_HOST / REM_PORT / REM_USER / REM_PASS
"""
import os
import sys

import paramiko

# Windows 控制台默认 GBK，远端输出里有 UTF-8/控制字符时会炸；统一改成 utf-8 + replace
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                    # noqa: BLE001
        pass

HOST = os.environ.get("REM_HOST", "connect.westb.seetacloud.com")
PORT = int(os.environ.get("REM_PORT", "38451"))
USER = os.environ.get("REM_USER", "root")
PASS = os.environ.get("REM_PASS", "")
TIMEOUT = int(os.environ.get("REM_TIMEOUT", "1800"))


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS,
              timeout=30, banner_timeout=30, auth_timeout=30,
              look_for_keys=False, allow_agent=False)
    return c


def run(cmd, timeout=None):
    c = connect()
    try:
        _in, out, err = c.exec_command(cmd, timeout=timeout or TIMEOUT,
                                       get_pty=False)
        o = out.read().decode("utf-8", "replace")
        e = err.read().decode("utf-8", "replace")
        rc = out.channel.recv_exit_status()
        return rc, o, e
    finally:
        c.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mode = sys.argv[1]
    if mode == "run":
        rc, o, e = run(sys.argv[2])
        if o:
            sys.stdout.write(o if o.endswith("\n") else o + "\n")
        if e:
            sys.stderr.write(e if e.endswith("\n") else e + "\n")
        print("[rc=%d]" % rc)
        return 0 if rc == 0 else 1
    c = connect()
    try:
        sftp = c.open_sftp()
        if mode == "put":
            sftp.put(sys.argv[2], sys.argv[3])
            print("[put ok] %s -> %s" % (sys.argv[2], sys.argv[3]))
        elif mode == "get":
            sftp.get(sys.argv[2], sys.argv[3])
            print("[get ok] %s -> %s" % (sys.argv[2], sys.argv[3]))
        else:
            print("未知模式:", mode)
            return 2
        sftp.close()
    finally:
        c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
