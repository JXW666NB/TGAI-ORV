# -*- coding: utf-8 -*-
"""全归约改两遍式（大张量多 block，小张量保持原路径逐位不变）。

实测依据（P100，21M 元素，纯前向）：ORV 9.881 ms（8.5 GB/s）vs torch 1.285 ms（65 GB/s）。
根因：cuda.py 的 sum_all 只启动 1 个 block（`(1,1,1)`），而 P100 有 56 个 SM。

本工具用**行号定位**而不是文本锚点 —— 之前两次改内核都栽在锚点上（0.9.74 的
`dK is undefined`、上一轮的"锚点不唯一"）。
"""
import io
import sys

P = "python/tgai_go/train/cuda.py"
KERNEL_LINE = 'extern "C" __global__ void reduce_sum(const float* a, float* out, int n) {'

NEW_KERNEL = [
    '',
    '/* 多 block 部分归约。以前全归约只启 1 个 block（56 个 SM 只用一个），21M 元素实测',
    '   8.5 GB/s，torch 约 65 GB/s。这里 G 个 block 各归约一段**连续**区间写入 part[G]',
    '   （连续切分保证合并访存），再由 reduce_sum 把 part[G] 归约成标量。 */',
    'extern "C" __global__ void reduce_sum_partial(const float* a, float* part, int n) {',
    '    __shared__ float smem[256];',
    '    const int nb = (int)gridDim.x;',
    '    const int bid = (int)blockIdx.x;',
    '    const int chunk = (n + nb - 1) / nb;',
    '    const int lo = bid * chunk;',
    '    int hi = lo + chunk;',
    '    if (hi > n) hi = n;',
    '    float local = 0.f;',
    '    for (int i = lo + (int)threadIdx.x; i < hi; i += (int)blockDim.x) {',
    '        local += a[i];',
    '    }',
    '    smem[threadIdx.x] = local;',
    '    __syncthreads();',
    '    for (int s = (int)blockDim.x >> 1; s > 0; s >>= 1) {',
    '        if ((int)threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];',
    '        __syncthreads();',
    '    }',
    '    if (threadIdx.x == 0) part[bid] = smem[0];',
    '}',
]

NEW_LAUNCH = [
    '        if n > (1 << 20):',
    '            # 大张量：两遍式。每块至少约 64K 元素，块数上限 1024。',
    '            nb = min(1024, (n + 65535) // 65536)',
    '            part = self.empty((nb,), dtype=np.float32)',
    '            self.launch("reduce_sum_partial", (nb, 1, 1), (256, 1, 1), [',
    '                self._ptr(buf), self._ptr(part), (n, ctypes.c_int),',
    '            ])',
    '            self.launch("reduce_sum", (1, 1, 1), (256, 1, 1), [',
    '                self._ptr(part), self._ptr(out), (nb, ctypes.c_int),',
    '            ])',
    '            return out',
    '        self.launch("reduce_sum", (1, 1, 1), (256, 1, 1), [',
    '            self._ptr(buf), self._ptr(out), (n, ctypes.c_int),',
    '        ])',
]


def main():
    text = io.open(P, encoding="utf-8").read()
    if "reduce_sum_partial" in text:
        print("已经改过")
        return 0
    L = text.split("\n")

    # ① 内核：在 reduce_sum 的定义块结束后插入
    ki = [i for i, l in enumerate(L) if l == KERNEL_LINE]
    assert len(ki) == 1, "reduce_sum 定义行数 = %d" % len(ki)
    k = ki[0]
    ke = [j for j in range(k, len(L)) if L[j] == "}"]
    assert ke, "找不到 reduce_sum 的收尾大括号"
    kend = ke[0]
    print("内核: 定义行 %d，收尾行 %d" % (k + 1, kend + 1))

    # ② 启动端：sum_all 里的 launch 三行
    si = [i for i, l in enumerate(L) if l.strip().startswith("def sum_all")]
    assert len(si) == 1, "sum_all 定义数 = %d" % len(si)
    s0 = si[0]
    lj = [j for j in range(s0, min(s0 + 40, len(L))) if 'self.launch("reduce_sum"' in L[j]]
    assert len(lj) == 1, "sum_all 内 reduce_sum 启动行 = %d" % len(lj)
    j = lj[0]
    # 该 launch 占三行：call / 参数 / '])'
    assert L[j + 2].strip() == "])", "启动块结尾不是 ]): %r" % L[j + 2]
    print("启动: 行 %d..%d" % (j + 1, j + 3))

    out = L[:kend + 1] + NEW_KERNEL + L[kend + 1:j] + NEW_LAUNCH + L[j + 3:]
    io.open(P, "w", encoding="utf-8", newline="\n").write("\n".join(out))
    import ast
    ast.parse(io.open(P, encoding="utf-8").read())
    print("已改：n>1M 两遍式，其余单 block 不变")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
