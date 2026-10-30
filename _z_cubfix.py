import io, py_compile

# 1) pyproject：加 [cuda] 可选依赖 + 版本 0.1.1
p = "pyproject.toml"
s = io.open(p, encoding="utf-8").read()
s = s.replace('version = "0.1.0"', 'version = "0.1.1"', 1)
s = s.replace('dependencies = ["numpy>=1.21"]',
'''dependencies = ["numpy>=1.21"]

# 裸机器（没装 CUDA Toolkit、也没有 torch）时，用
#     pip install "tgai-orv[cuda]"
# 把 cuBLAS / NVRTC / CUDA runtime 装进同一个 venv；我们的加载器会自动搜
# site-packages/nvidia/*/lib。已经有系统 CUDA 或 torch 的机器不需要这一项。
[project.optional-dependencies]
cuda = [
  "nvidia-cuda-runtime-cu12>=12.1; sys_platform == 'linux'",
  "nvidia-cublas-cu12>=12.1; sys_platform == 'linux'",
  "nvidia-cuda-nvrtc-cu12>=12.1; sys_platform == 'linux'",
]''', 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)

# 2) 报错里写清修法
p2 = "python/tgai_go/train/cuda.py"
s2 = io.open(p2, encoding="utf-8").read()
old = '''            raise CudaError(
                    "找不到 cublas64_*.dll。已查目录: %s" % (self._dirs or ["<空>"])
                )'''
if old not in s2:
    old = '''raise CudaError(
                    "找不到 cublas64_*.dll。已查目录: %s" % (self._dirs or ["<空>"])
                )'''
assert old in s2
s2 = s2.replace(old, '''raise CudaError(
                    "找不到 cublas64_*.dll。已查目录: %s\\n"
                    "修法：装 CUDA Toolkit 并设 CUDA_PATH，或用 TGAI_CUDA_BIN 指向含 cublas64_*.dll 的目录"
                    % (self._dirs or ["<空>"])
                )''', 1)

old2 = '"找不到 libcublas.so.12。已查目录: %s"'
assert old2 in s2
s2 = s2.replace(old2, '"找不到 libcublas.so.12。已查目录: %s\\n"\n                    "修法：pip install \\"tgai-orv[cuda]\\"（裸机器），或设 TGAI_CUDA_BIN 指向含 libcublas.so.12 的目录"', 1)
io.open(p2, "w", encoding="utf-8").write(s2)
py_compile.compile(p2, cfile=__import__("tempfile").mktemp(), doraise=True)
print("已改：pyproject 加 [cuda] extra + 版本 0.1.1；cublas 报错补修法")
