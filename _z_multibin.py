import io, py_compile, tempfile
p = r"python/tgai_go/train/cuda.py"
s = io.open(p, encoding="utf-8").read()

old = '''    extra = os.environ.get("TGAI_CUDA_BIN")
    if extra:
        dirs.append(extra)'''
new = '''    # TGAI_CUDA_BIN 支持多路径（os.pathsep 分隔）：裸机器上 cuBLAS / NVRTC / CUDA runtime
    # 往往分散在 nvidia/*/lib 三个目录里，只给一个会找不到另一个。
    extra = os.environ.get("TGAI_CUDA_BIN")
    if extra:
        for part in extra.split(os.pathsep):
            part = part.strip()
            if part:
                dirs.append(part)'''
n = s.count(old)
assert n == 2, "期望 2 处（windows + linux），实际 %d" % n
s = s.replace(old, new)

s = s.replace('version = "0.1.1"', 'version = "0.1.2"')
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
py_compile.compile(p, cfile=tempfile.mktemp(), doraise=True)

pp = "pyproject.toml"
t = io.open(pp, encoding="utf-8").read().replace('version = "0.1.1"', 'version = "0.1.2"', 1)
io.open(pp, "w", encoding="utf-8", newline="\n").write(t)
print("TGAI_CUDA_BIN 已支持多路径；版本 0.1.2")
