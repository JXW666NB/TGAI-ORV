# -*- coding: utf-8 -*-
"""部署 v4 能力版引擎 DLL（并可把产线模型指向 v4），**任一步失败自动回滚**。

为什么需要单独一个脚本（实测教训，2026-11-09）：

    只把 launcher 的 MDL 改成 v4、不换引擎 DLL ⇒ 服务起不来：
        OSError: exception: access violation writing 0x0000000000000024
    原因：旧 `libtgai_engine_gpu.dll`（2026-10-19 构建）不认识 version=4，
    CUDA 加载路径按 v3 布局取数直接越界访问。
    ⇒ **模型升 v4 必须同时部署支持 v4 的引擎 DLL**。

顺序与安全：
    备份 DLL（带时间戳，sha256 入库） → 替换 → 停 11450/11451 → 等显存回落
    → 先起 11450（需 13.7 GB 连续空闲）→ 冒烟 → 再起 11451
    → 任何一步失败：DLL 回滚 + launcher 回滚 + 重启 v3

用法：
    python deploy_v4_engine.py --dry-run
    python deploy_v4_engine.py --apply --dll <新的 v4 能力版 dll>
    python deploy_v4_engine.py --rollback           # 回到备份的 DLL 与 v3 模型
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

DLL = r"D:\TGAI\libtgai_engine_gpu.dll"
LAUNCH = {11450: r"D:\tgai_orv\launch11450.py",
          11451: r"D:\tgai_orv\launch11451.py"}
BAK_SUFFIX = ".bak_before_v4"
V3 = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7.tg"
V4 = r"D:\TGAI\models\V2.7\TGAI_NB_V2.7_v4.tg"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


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


def stop_all():
    for t in ("TGAI-GO-11450", "TGAI-GO-11451"):
        sh('schtasks /end /tn "%s"' % t)
    for port in (11450, 11451):
        o = sh("netstat -ano | findstr :%d | findstr LISTENING" % port).strip()
        if o:
            sh("taskkill /F /PID %s" % o.split()[-1])


def start(port, budget=300):
    subprocess.Popen([r"C:\Python311\python.exe", "-u", LAUNCH[port]],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < budget:
        time.sleep(5)
        if port_up(port):
            return True
    return False


def smoke():
    """短问答冒烟 + 乱码判定。"""
    body = {"model": "tgai-nb-v2.7",
            "prompt": "<\u7528\u6237>\n\u4f60\u597d\n<TGAI>",
            "stream": False, "max_tokens": 24, "temperature": 0.3}
    req = urllib.request.Request("http://127.0.0.1:11450/v1/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json; charset=utf-8"},
                                 method="POST")
    o = json.loads(urllib.request.urlopen(req, timeout=180).read())
    txt = o["choices"][0]["text"]
    return txt, (o.get("usage") or {}).get("completion_tokens")


def launcher_model(path):
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith("MDL ") and "=" in line:
            return line.split("=", 1)[1].strip().strip('"').lstrip("r")
    return None


def set_launcher_model(path, model):
    src = open(path, encoding="utf-8", errors="replace").read().splitlines()
    out, hit = [], False
    for line in src:
        if line.startswith("MDL ") and "=" in line:
            out.append('MDL = r"%s"' % model)
            hit = True
        else:
            out.append(line)
    if not hit:
        raise SystemExit("launcher 里找不到 MDL 行：%s（修法：确认该文件仍是 launch 脚本）" % path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")


def newest_dll_backup():
    c = sorted(glob.glob(DLL + ".bak_v3_*"))
    return c[-1] if c else None


def do_rollback():
    stop_all()
    b = newest_dll_backup()
    if b:
        shutil.copy2(b, DLL)
        print("已回滚 DLL ← %s" % os.path.basename(b))
    for p in LAUNCH.values():
        bak = p + BAK_SUFFIX
        if os.path.exists(bak):
            shutil.copy2(bak, p)
            print("已回滚 %s" % os.path.basename(p))
    ok = start(11450)
    ok2 = start(11451) if ok else False
    print("回滚后 11450=%s 11451=%s GPU=%d" % (ok, ok2, vram()))
    return 0 if (ok and ok2) else 3


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--rollback", action="store_true")
    ap.add_argument("--dll", default=r"D:\v4build\out_gpu3\libtgai_engine_gpu.dll")
    args = ap.parse_args()

    print("=== 当前状态 ===")
    print("  DLL=%s %d B sha=%s" % (os.path.basename(DLL), os.path.getsize(DLL), sha(DLL)[:16]))
    for p in LAUNCH.values():
        print("  %-16s MDL=%s" % (os.path.basename(p), launcher_model(p)))
    print("  11450=%s 11451=%s GPU=%d MiB" % (port_up(11450), port_up(11451), vram()))
    print("=== 目标 ===")
    print("  DLL ← %s（%d B，sha=%s）"
          % (args.dll, os.path.getsize(args.dll) if os.path.exists(args.dll) else -1,
             sha(args.dll)[:16] if os.path.exists(args.dll) else "?"))
    print("  MDL → %s" % V4)

    if args.rollback:
        return do_rollback()
    if not args.apply:
        print("\n[dry-run] 未做任何修改。加 --apply 执行。")
        return 0

    if not os.path.exists(args.dll):
        raise SystemExit("找不到新 DLL：%s（修法：先按 build_cuda_v100.bat 配方编译再部署）" % args.dll)
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = DLL + ".bak_v3_" + ts
    shutil.copy2(DLL, bak)
    print("已备份 DLL → %s（sha=%s）" % (os.path.basename(bak), sha(bak)[:16]))
    shutil.copy2(args.dll, DLL)
    print("已替换 DLL（sha=%s）" % sha(DLL)[:16])

    for p in LAUNCH.values():
        b = p + BAK_SUFFIX
        if not os.path.exists(b):
            shutil.copy2(p, b)          # 首次才备份，保留最初的 v3 版本
    for p in LAUNCH.values():
        set_launcher_model(p, V4)
    print("已把两个 launcher 的 MDL 指向 v4")

    stop_all()
    for _ in range(30):
        time.sleep(5)
        if vram() < 2000:
            break
    print("显存 %d MiB" % vram())

    if not start(11450):
        print("!! 11450 未监听")
        return do_rollback()
    print("11450 已监听，GPU=%d MiB" % vram())
    try:
        txt, tok = smoke()
        print("冒烟 tok=%s 文本=%r" % (tok, txt[:80]))
        if not txt.strip() or "\u951f" in txt:
            print("!! 冒烟输出异常")
            return do_rollback()
    except Exception as e:
        print("!! 冒烟异常 %s %s" % (type(e).__name__, str(e)[:120]))
        return do_rollback()
    if not start(11451):
        print("!! 11451 未监听")
        return do_rollback()
    print("11451 已监听，GPU=%d MiB" % vram())
    print("=== 完成：DLL 支持 v4，模型 v4，两服务在线 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
