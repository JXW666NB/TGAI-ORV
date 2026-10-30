import io
p = "tools/accept.py"
s = io.open(p, encoding="utf-8").read()
old = '''    shapes = [("D256 NH8 FF1280 L8 S64", dict(D=256, NH=8, FF=1280, L=8, MAXLEN=64)),
              ("D128 NH4 FF512  L4 S64", dict(D=128, NH=4, FF=512, L=4, MAXLEN=64)),
              ("D512 NH8 FF2048 L2 S64", dict(D=512, NH=8, FF=2048, L=2, MAXLEN=64)),
              ("D192 NH6 FF768  L3 S96", dict(D=192, NH=6, FF=768, L=3, MAXLEN=96))]'''
assert old in s, "形状列表未匹配"
new = '''    # 覆盖 6 组（L1 判据要求 ≥6），含边界：非对齐 FF=1000、极短序列 S=8、深网络 L=8
    shapes = [("D256 NH8 FF1280 L8 S64", dict(D=256, NH=8, FF=1280, L=8, MAXLEN=64)),
              ("D128 NH4 FF512  L4 S64", dict(D=128, NH=4, FF=512, L=4, MAXLEN=64)),
              ("D512 NH8 FF2048 L2 S64", dict(D=512, NH=8, FF=2048, L=2, MAXLEN=64)),
              ("D192 NH6 FF768  L3 S96", dict(D=192, NH=6, FF=768, L=3, MAXLEN=96)),
              ("D256 NH8 FF1000 L2 S64", dict(D=256, NH=8, FF=1000, L=2, MAXLEN=64)),
              ("D128 NH4 FF512  L2 S8 ", dict(D=128, NH=4, FF=512, L=2, MAXLEN=8))]'''
s = s.replace(old, new, 1)
s = s.replace('rows.append(("形状矩阵 4 组"', 'rows.append(("形状矩阵 6 组"')
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("accept.py 形状扩到 6 组")
