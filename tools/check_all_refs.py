# -*- coding: utf-8 -*-
"""校验 COMMUNITY_ONBOARDING.md 与 CAPABILITY_AUDIT.md 引用的文档都存在。"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 已知缺失且**已在文档中标注**的引用（不报错）。
#: 这 9 个是 TGAI GO（推理侧）的工具，建 ORV 仓库时未带入；
#: docs/INDEX.md 已在相应表格前加了说明。要恢复请从 GO 侧移植。
KNOWN_MISSING = {
    "tools/verify_bpe_v27.py", "tools/verify_bf16_v2.py",
    "tools/verify_bf16_direct.py", "tools/verify_q4k_cross.py",
    "tools/verify_twolevel_fixed.py", "tools/pt_reference_bf16.py",
    "tools/compare_layers.py", "tools/esp32_budget2.py",
    "tools/bench_q4k.c",
}

bad = []
skipped = []
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
            if ref in KNOWN_MISSING:
                skipped.append(ref)
            else:
                bad.append("%s -> %s" % (name, ref))

print("缺失:", sorted(bad) or "无")
print("已知缺失（已标注，跳过）:", len(set(skipped)), "个")
sys.exit(1 if bad else 0)
