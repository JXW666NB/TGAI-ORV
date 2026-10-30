import io, py_compile, tempfile
p = r"python/tgai_go/train/cuda.py"
s = io.open(p, encoding="utf-8").read()

anchor = '''    dirs.append("/usr/lib/x86_64-linux-gnu")
    dirs.append("/usr/lib64")
    return _dedupe_dirs(dirs)'''
assert anchor in s
new = '''    # 自动发现：本机任何 Python 环境里的 nvidia/*/lib（很常见的情形是
    # 系统里装了 torch，CUDA 库随它一起进来，而我们跑在一个干净的 venv 里）。
    # 这样 `pip install tgai-orv` 在"机器上有 torch 但 venv 里没有"时也能直接用。
    cands = []
    for entry in list(sys.path):
        if entry:
            cands.append(os.path.join(entry, "nvidia"))
    cands += glob.glob("/usr/local/lib/python3*/site-packages/nvidia")
    cands += glob.glob("/usr/local/lib/python3*/dist-packages/nvidia")
    cands += glob.glob("/usr/lib/python3*/site-packages/nvidia")
    cands += glob.glob(os.path.expanduser("~/.local/lib/python3*/site-packages/nvidia"))
    for nv in cands:
        if not os.path.isdir(nv):
            continue
        try:
            pkgs = sorted(os.listdir(nv))
        except OSError:
            continue
        for pkg in pkgs:
            for sub in ("lib", "lib64"):
                d = os.path.join(nv, pkg, sub)
                if os.path.isdir(d):
                    dirs.append(d)
    dirs.append("/usr/lib/x86_64-linux-gnu")
    dirs.append("/usr/lib64")
    return _dedupe_dirs(dirs)'''
s = s.replace(anchor, new, 1)
if "\nimport glob" not in s:
    s = s.replace("\nimport ctypes", "\nimport glob\nimport ctypes", 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
py_compile.compile(p, cfile=tempfile.mktemp(), doraise=True)

pp = "pyproject.toml"
t = io.open(pp, encoding="utf-8").read().replace('version = "0.1.2"', 'version = "0.1.3"', 1)
io.open(pp, "w", encoding="utf-8", newline="\n").write(t)
print("已加自动发现系统 nvidia 库；版本 0.1.3")
