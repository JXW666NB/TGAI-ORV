# -*- coding: utf-8 -*-
"""产线体检 + 自愈：把 11450/11451 拉回到监听状态（幂等，可反复跑）。

为什么需要（2026-11-09 真实事故）：

    评测脚本 `v27_eval_real.py` / `v27_next_token.py` 的 `isolate()` 会把
    `launch11450.py` / `launch11451.py` **改名成 `.off`** 来防止保活任务把服务拉起来。
    我跑评测时**没加 `--restore`**，于是 launcher 一直是 `.off`；
    随后的启动脚本 Popen 一个不存在的路径 ⇒ **静默失败**，产线白白停了十几分钟。

所以自愈顺序必须是：
    ① 把 `*.py.off` 改回 `*.py`
    ② 启用并运行计划任务（TGAI-GO-11450 / 11451）
    ③ 端口没起来就**直接** Popen launcher（先 11450，就绪后再 11451）
    ④ 最后核对端口 + 显存

用法：
    python ensure_production.py            # 体检+修复
    python ensure_production.py --check    # 只看不改
"""
import argparse
import glob
import os
import socket
import subprocess
import sys
import time

LAUNCH = {11450: r"D:\tgai_orv\launch11450.py",
          11451: r"D:\tgai_orv\launch11451.py"}
TASKS = {11450: "TGAI-GO-11450", 11451: "TGAI-GO-11451"}


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True,
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


def unrename():
    """把 isolate() 留下的 `.off` 改回来。"""
    fixed = []
    for pat in (r"D:\tgai_orv\launch*.py.off",):
        for p in glob.glob(pat):
            dst = p[:-4]
            if os.path.exists(dst):
                os.remove(p)
                fixed.append(dst + "（删除重复 .off）")
            else:
                os.rename(p, dst)
                fixed.append(dst)
    return fixed


def start(port, budget=300):
    if not os.path.exists(LAUNCH[port]):
        print("   !! launcher 不存在：%s" % LAUNCH[port])
        return False
    subprocess.Popen([r"C:\Python311\python.exe", "-u", LAUNCH[port]],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < budget:
        time.sleep(5)
        if port_up(port):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    print("=== 体检 ===")
    print("  11450=%s 11451=%s GPU=%d MiB" % (port_up(11450), port_up(11451), vram()))
    offs = glob.glob(r"D:\tgai_orv\launch*.py.off")
    print("  遗留 .off：%s" % ([os.path.basename(p) for p in offs] or "无"))
    if args.check:
        return 0 if (port_up(11450) and port_up(11451)) else 1

    print("=== 修复 ===")
    for f in unrename():
        print("  已改回 %s" % f)
    for port in (11450, 11451):
        t = TASKS[port]
        sh('schtasks /change /tn "%s" /enable' % t)
        if not port_up(port):
            sh('schtasks /run /tn "%s"' % t)
            for _ in range(24):
                time.sleep(5)
                if port_up(port):
                    break
        if not port_up(port):
            print("   任务没起来，直接启动 %d" % port)
            start(port)
        print("   %d=%s GPU=%d MiB" % (port, port_up(port), vram()))
        if port == 11450 and not port_up(11450):
            print("   !! 11450 起不来（需要 13.7 GB 连续空闲），先修显存再重试")
            return 3
    ok = port_up(11450) and port_up(11451)
    print("=== 结果：%s（GPU %d MiB）===" % ("正常" if ok else "异常", vram()))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
