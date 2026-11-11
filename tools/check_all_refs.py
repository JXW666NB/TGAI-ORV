# -*- coding: utf-8 -*-
"""校验 COMMUNITY_ONBOARDING.md 与 CAPABILITY_AUDIT.md 引用的文档都存在。"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bad = []
for name in ("COMMUNITY_ONBOARDING.md", "CAPABILITY_AUDIT.md", "DESIGN.md",
             "PLUGIN_CHARTER.md", "INDEX.md"):
    p = os.path.join(ROOT, "docs", name)
    if not os.path.isfile(p):
        bad.append("%s 不存在" % name)
        continue
    s = io.open(p, encoding="utf-8").read()
    for ref in set(re.findall(r"`(docs/[\w./\-]+)`", s)):
        if not os.path.isfile(os.path.join(ROOT, ref)):
            bad.append("%s -> %s" % (name, ref))
    for ref in set(re.findall(r"`(tools/[\w./\-]+)`", s)):
        if not os.path.isfile(os.path.join(ROOT, ref)):
            bad.append("%s -> %s" % (name, ref))

print("缺失:", bad or "无")
sys.exit(1 if bad else 0)
