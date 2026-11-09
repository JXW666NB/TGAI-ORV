# -*- coding: utf-8 -*-
"""把产线 11450/11451 切到 v4 模型文件（**可回滚**），支持干跑。

为什么要脚本而不是手工：这台机器上 11450 需要约 13.7 GB 连续空闲显存，
必须先起 11450 再起 11451（实测：顺序反了两者会互相抢显存，后者 OOM）。

用法：
    python switch_v4.py --dry-run        # 只打印将要做什么（默认）
    python switch_v4.py --apply          # 真正执行
    python switch_v4.py --rollback       # 回滚到 v3
"""
import argparse
import os
import shutil
import socket
import subprocess
import sys
import time

LAUNCHERS = (r"D:\tgai_orv\launch11450.py", r"D:\tgai_orv\launch11451.py")
V3 = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7.tg"
V4 = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7_v4.tg"
BAK_SUFFIX = ".bak_before_v4"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True,
                          encoding="utf-8", errors="ignore").stdout


def port_up(p):
    s = socket.socket()
    s.settimeout(1.0)
    try:
        s.connect(("127.0.0.1", p))
        return True
    except Exception:
        return False
    finally:
        s.close()


def vram():
    try:
        return int(sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader").split()[0])
    except Exception:
        return -1


def launcher_model(path):
    """读出 launcher 里当前的 MDL 指向。"""
    try:
        for line in open(path, encoding="utf-8", errors="ignore"):
            if line.startswith("MDL "):
                return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return None


def set_launcher_model(path, new_model):
    """把 MDL 行替换成新路径（自动备份一次）。"""
    src = open(path, encoding="utf-8", errors="replace").read()
    out = []
    changed = False
    for line in src.splitlines():
        if line.startswith("MDL ") and "=" in line:
            out.append('MDL = r"%s"' % new_model)
            changed = True
        else:
            out.append(line)
    if not changed:
        raise SystemExit("launcher 里找不到 MDL 行：%s" % path)
    bak = path + BAK_SUFFIX
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    return bak


def stop_all():
    for t in ("TGAI-GO-11450", "TGAI-GO-11451"):
        sh('schtasks /end /tn "%s"' % t)
    for port in (11450, 11451):
        o = sh("netstat -ano | findstr :%d | findstr LISTENING" % port).strip()
        if o:
            sh("taskkill /F /PID %s" % o.split()[-1])


def start_and_wait(port, launcher, budget=240):
    """先起一个，等到它监听为止（避免两个同时抢显存）。"""
    subprocess.Popen([r"C:\Python311\python.exe", "-u", launcher],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < budget:
        time.sleep(5)
        if port_up(port):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="只打印计划（默认）")
    g.add_argument("--apply", action="store_true", help="执行切换")
    g.add_argument("--rollback", action="store_true", help="回滚到 v3")
    args = ap.parse_args()
    target = V4 if not args.rollback else V3

    print("=== 当前状态 ===")
    for p in LAUNCHERS:
        print("  %-34s MDL=%s" % (os.path.basename(p), launcher_model(p)))
    print("  11450=%s 11451=%s GPU=%d MiB" % (port_up(11450), port_up(11451), vram()))
    print("=== 目标 ===")
    print("  把所有 launcher 的 MDL 指向：%s" % target)
    print("  步骤：备份 launcher → 改 MDL → 停两个服务 → 等显存回落 → 先起 11450 → 再起 11451")

    if not (args.apply or args.rollback):
        print("\n[dry-run] 未做任何修改。加 --apply 才会执行。")
        return 0

    for p in LAUNCHERS:
        bak = set_launcher_model(p, target)
        print("  已改 %s（备份 %s）" % (os.path.basename(p), os.path.basename(bak)))

    stop_all()
    for _ in range(24):
        time.sleep(5)
        if vram() < 2000:
            break
    print("  显存回落到 %d MiB" % vram())

    ok50 = start_and_wait(11450, LAUNCHERS[0])
    print("  11450=%s" % ok50)
    ok51 = start_and_wait(11451, LAUNCHERS[1]) if ok50 else False
    print("  11451=%s" % ok51)
    print("  最终 GPU=%d MiB" % vram())
    if not (ok50 and ok51):
        print("!! 启动失败，请用 --rollback 回滚")
        return 3
    print("切换完成。回滚：python switch_v4.py --rollback")
    return 0


if __name__ == "__main__":
    sys.exit(main())
