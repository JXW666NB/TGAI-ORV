import io, py_compile, tempfile
p = r"python/tgai_go/train/cuda.py"
L = io.open(p, encoding="utf-8").read().split("\n")
# 用行号精确重写 2299-2304（1-based）与 2307-2311
good_win = [
    "            if path is None:",
    "                raise CudaError(",
    "                    \"找不到 cublas64_*.dll。已查目录: %s\\n\"",
    "                    \"修法：装 CUDA Toolkit 并设 CUDA_PATH，或用 TGAI_CUDA_BIN 指向含 cublas64_*.dll 的目录\"",
    "                    % (self._dirs or [\"<空>\"])",
    "                )",
]
L[2298:2304] = good_win            # 索引 2298..2303（第 2299..2304 行）
good_linux = [
    "        if path is None:",
    "            raise CudaError(",
    "                \"找不到 libcublas.so.12。已查目录: %s\\n\"",
    "                \"修法：pip install 'tgai-orv[cuda]'（裸机器），或设 TGAI_CUDA_BIN 指向含 libcublas.so.12 的目录\"",
    "                % (self._dirs or [\"<空>\"])",
    "            )",
]
# 找当前的 linux 块：从 'if path is None:' 到 'return path' 之间的行
i = next(k for k in range(2290, 2330) if L[k].strip() == 'if path is None:' and 'libcublas' in "\n".join(L[k:k+6]))
j = next(k for k in range(i, i+8) if L[k].strip() == 'return path')
L[i:j] = good_linux
s = "\n".join(L)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
py_compile.compile(p, cfile=tempfile.mktemp(), doraise=True)
print("cuda.py 缩进已修复，语法 OK")
