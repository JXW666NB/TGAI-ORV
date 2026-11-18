# -*- coding: utf-8 -*-
"""租机远控：用 paramiko 跑命令 / 传文件（密码从 %TEMP%\\orv_rent.json 读，不进仓库）。

用法：
    python lab/rent/rsh.py --cmd "nvidia-smi"
    python lab/rent/rsh.py --put lab/rent/make_packed.py /root/make_packed.py
    python lab/rent/rsh.py --get /root/out.txt ./out.txt
    python lab/rent/rsh.py --cmd "python3 x.py" --timeout 3600

凭据文件格式（放在 %TEMP%\\orv_rent.json）：
    {"host": "...", "port": 22659, "user": "root", "password": "..."}
"""
import argparse
import json
import os
import sys
import tempfile
import time

import paramiko


def creds():
    path = os.path.join(tempfile.gettempdir(), "orv_rent.json")
    if not os.path.exists(path):
        sys.exit("缺少凭据文件：%s（应含 host/port/user/password）" % path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def connect(c, timeout=30):
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(c["host"], port=int(c["port"]), username=c["user"],
                password=c["password"], timeout=timeout,
                allow_agent=False, look_for_keys=False)
    return cli


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd")
    ap.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"))
    ap.add_argument("--get", nargs=2, metavar=("REMOTE", "LOCAL"))
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args()
    c = creds()
    cli = connect(c)
    try:
        if a.put:
            sftp = cli.open_sftp()
            sftp.put(a.put[0], a.put[1])
            print("[put] %s -> %s" % (a.put[0], a.put[1]))
            sftp.close()
        if a.get:
            sftp = cli.open_sftp()
            sftp.get(a.get[0], a.get[1])
            print("[get] %s -> %s" % (a.get[0], a.get[1]))
            sftp.close()
        if a.cmd:
            stdin, stdout, stderr = cli.exec_command(a.cmd, timeout=a.timeout,
                                                    get_pty=True)
            t0 = time.time()
            # ★ 按**字节**读再解码：远端可能吐出非 UTF-8（vulkaninfo 的本地化/二进制输出），
            #   文本模式会抛 UnicodeDecodeError 把整条命令废掉（实测踩过）。
            while True:
                chunk = stdout.channel.recv(65536)
                if not chunk:
                    break
                sys.stdout.write(chunk.decode("utf-8", "replace"))
                sys.stdout.flush()
                if time.time() - t0 > a.timeout:
                    print("[timeout] 超过 %.0f s，停止读取" % a.timeout)
                    break
            rc = stdout.channel.recv_exit_status()
            err = stderr.read().decode("utf-8", "replace")
            if err.strip():
                sys.stderr.write(err)
            print("[exit] %d" % rc)
            sys.exit(0 if rc == 0 else 1)
    finally:
        cli.close()


if __name__ == "__main__":
    main()
