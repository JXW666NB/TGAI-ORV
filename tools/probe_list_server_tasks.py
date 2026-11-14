# -*- coding: utf-8 -*-
"""列出所有会拉起 run_tgaigo_server 的计划任务（只读，基于已备份的 XML）。"""
import glob
import io
import os

KEYS = ("run_tgaigo_server", "tgaigo", "11450", "11451", "11452", "11453")


def main():
    for path in sorted(glob.glob(r"D:\_task_backup\*.xml")):
        try:
            txt = io.open(path, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        low = txt.lower()
        hit = [k for k in KEYS if k in low]
        if not hit:
            continue
        # 抽出 <Command> 与 <Arguments>
        import re
        cmd = re.search(r"<Command>(.*?)</Command>", txt, re.S)
        args = re.search(r"<Arguments>(.*?)</Arguments>", txt, re.S)
        name = os.path.basename(path)[:-4]
        c = (cmd.group(1).strip() if cmd else "")
        a = (args.group(1).strip() if args else "")
        print("%-28s | %s %s" % (name[:28], c[:80], a[:60]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
