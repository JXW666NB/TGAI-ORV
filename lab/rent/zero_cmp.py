# -*- coding: utf-8 -*-
"""ZeRO 逐位比对器：读 v3_pretrain 的 train_log*.jsonl，比较 --zero 0/1/2。

判据（任务书验收 A2）：
  1. 同一 rank 上，--zero 0/1/2 的 **loss 与 gnorm 每一步逐位相同**；
  2. 同一模式下，rank0 与 rank1 的 **gnorm 逐位相同**（loss 不同：各 rank 吃不同样本）。

退出码 0 = PASS，1 = FAIL。
"""
import json
import sys


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cmp_series(name, base, other):
    bad = []
    if len(base) != len(other):
        bad.append("%s: 步数不同 %d vs %d" % (name, len(base), len(other)))
    for i in range(min(len(base), len(other))):
        b, o = base[i], other[i]
        for k in ("step", "loss", "gnorm", "n_valid"):
            if b.get(k) != o.get(k):
                bad.append("%s step=%s 字段 %s 不同：%r vs %r"
                           % (name, b.get("step"), k, b.get(k), o.get(k)))
    return bad


def main(argv):
    pos = [a for a in argv if not a.startswith("--")]
    r1_at = argv.index("--r1") if "--r1" in argv else -1
    if r1_at >= 0:
        r0_paths = pos[:3]
        r1_paths = pos[3:6]
    else:
        r0_paths = pos[:3]
        r1_paths = []
    if len(r0_paths) != 3:
        print("用法: python zero_cmp.py <log_zero0> <log_zero1> <log_zero2> "
              "[--r1 <r1_zero0> <r1_zero1> <r1_zero2>]")
        return 2
    r0 = [load(p) for p in r0_paths]
    print("== rank0 逐步 (loss / gnorm) ==")
    n = max(len(x) for x in r0)
    for i in range(n):
        parts = []
        for z in range(3):
            if i < len(r0[z]):
                r = r0[z][i]
                parts.append("z%d L=%.17g G=%.17g" % (z, r["loss"], r["gnorm"]))
            else:
                parts.append("z%d ---" % z)
        print("step %2d  %s" % (i, "  |  ".join(parts)))

    bad = []
    bad += cmp_series("rank0 zero0->zero1", r0[0], r0[1])
    bad += cmp_series("rank0 zero0->zero2", r0[0], r0[2])

    if r1_paths:
        r1 = [load(p) for p in r1_paths]
        bad += cmp_series("rank1 zero0->zero1", r1[0], r1[1])
        bad += cmp_series("rank1 zero0->zero2", r1[0], r1[2])
        # gnorm 跨 rank 必须一致（归约后的范数是全局量）
        for z in range(3):
            a, b = r0[z], r1[z]
            for i in range(min(len(a), len(b))):
                if a[i]["gnorm"] != b[i]["gnorm"]:
                    bad.append("zero%d step=%s 的 gnorm 跨 rank 不同：%.17g vs %.17g"
                               % (z, a[i].get("step"), a[i]["gnorm"], b[i]["gnorm"]))

    if bad:
        print("\nBITWISE FAIL (%d 处不同)" % len(bad))
        for b in bad[:20]:
            print("  - " + b)
        return 1
    print("\nBITWISE PASS：zero 0/1/2 的 loss 与 gnorm 每一步逐位相同"
          + ("（rank0 + rank1，且跨 rank gnorm 一致）" if r1_paths else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
