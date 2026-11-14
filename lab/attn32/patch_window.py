# -*- coding: utf-8 -*-
"""lab #2：smQ/smDO 改成 16 维窗口。

## 原则：**只增不改**

`flash_qk_dp` 被 g2 变体（`attn_bwd_flash_g2`）共用，改它的签名会连带破坏 g2。
所以新增 `flash_qk_dp_win`，只让 64 行反向内核调用它；旧函数原样保留。

有同样代码的锚点（smem 切分、装载调用）一律用 `rep_first` —— 反向内核在 g2 之前。

用法：python lab/attn32/patch_window.py
"""
import io

CU = "lab/attn32/kernel.cu"

LOAD_WIN = """// ── 窗口装载：把 src 的 [d0, d0+nd) 维、tok0 起 BT 行装进 dst（布局 [dl][row]）──
__device__ __forceinline__ void load_win(const float* __restrict__ src, float* __restrict__ dst,
                                         int tok0, int h, int nh, int dh, int tid, int seq,
                                         int d0, int nd, int stride) {
#pragma unroll 1
    for (int e = tid; e < nd * BT; e += 256) {
        int local = e / nd;
        int dl = e - local * nd;
        int tok = tok0 + local;
        float v = 0.f;
        if (tok < seq) v = src[((tok * nh + h) * dh) + d0 + dl];
        dst[dl * stride + local] = v;
    }
}

"""

QKDP_WIN = """// 窗口版：把若干个 head_dim 窗口的 QK 与 dP 累加到调用者给的 sx/dp（不清零）。
__device__ __forceinline__ void flash_qk_dp_win(const float* __restrict__ smQ,
                                                const float* __restrict__ smDO,
                                                const float* __restrict__ smK,
                                                const float* __restrict__ smV,
                                                int row0, int lane, int d0,
                                                float sx[8], float dp[8]) {
#pragma unroll 8
    for (int dl = 0; dl < DCHUNK; ++dl) {
        int d = d0 + dl;
        float kv = smK[d * BC + lane];
        float vv = smV[d * BC + lane];
        float4 q0 = *reinterpret_cast<const float4*>(smQ + dl * QSTR + row0);
        float4 q1 = *reinterpret_cast<const float4*>(smQ + dl * QSTR + row0 + 4);
        float4 o0 = *reinterpret_cast<const float4*>(smDO + dl * QSTR + row0);
        float4 o1 = *reinterpret_cast<const float4*>(smDO + dl * QSTR + row0 + 4);
        sx[0] = fmaf(q0.x, kv, sx[0]); dp[0] = fmaf(o0.x, vv, dp[0]);
        sx[1] = fmaf(q0.y, kv, sx[1]); dp[1] = fmaf(o0.y, vv, dp[1]);
        sx[2] = fmaf(q0.z, kv, sx[2]); dp[2] = fmaf(o0.z, vv, dp[2]);
        sx[3] = fmaf(q0.w, kv, sx[3]); dp[3] = fmaf(o0.w, vv, dp[3]);
        sx[4] = fmaf(q1.x, kv, sx[4]); dp[4] = fmaf(o1.x, vv, dp[4]);
        sx[5] = fmaf(q1.y, kv, sx[5]); dp[5] = fmaf(o1.y, vv, dp[5]);
        sx[6] = fmaf(q1.z, kv, sx[6]); dp[6] = fmaf(o1.z, vv, dp[6]);
        sx[7] = fmaf(q1.w, kv, sx[7]); dp[7] = fmaf(o1.w, vv, dp[7]);
    }
}

"""


def rep(s, old, new, tag, count=1):
    n = s.count(old)
    if n != count:
        raise SystemExit("锚点 %s 命中 %d 次（期望 %d）：\n%s" % (tag, n, count, old[:140]))
    print("  改 %-18s (%d 处)" % (tag, n))
    return s.replace(old, new, count)


def rep_first(s, old, new, tag):
    n = s.count(old)
    if n < 1:
        raise SystemExit("锚点 %s 未命中：\n%s" % (tag, old[:140]))
    print("  改 %-18s (共 %d 处，只改第一处 = 反向内核)" % (tag, n))
    return s.replace(old, new, 1)


def main():
    s = io.open(CU, encoding="utf-8").read()

    s = rep(s, "#define QSTR (BT + 4)",
            "#define DCHUNK 16            // head_dim 窗口宽度（CUTLASS: ThreadK*2）\n#define QSTR (BT + 4)",
            "DCHUNK")

    # load_win 与 flash_qk_dp_win 都插在 load_T72 之前
    s = rep(s, "__device__ __forceinline__ void load_T72(",
            LOAD_WIN + QKDP_WIN + "__device__ __forceinline__ void load_T72(", "新函数")

    s = rep_first(s, """    float* smQ = smem;
    float* smDO = smQ + 64 * QSTR;
    float* smK = smDO + 64 * QSTR;""",
                  """    float* smQ = smem;                     // [DCHUNK][QSTR] 窗口
    float* smDO = smQ + DCHUNK * QSTR;     // [DCHUNK][QSTR] 窗口
    float* smK = smDO + DCHUNK * QSTR;     // [DH][BC] 全维""", "smem切分")

    s = rep_first(s, """    load_T72(Q, smQ, t0, h, nh, dh, tid, seq);
    load_T72(dO, smDO, t0, h, nh, dh, tid, seq);
    __syncthreads();
""", "", "移除首载")

    s = rep_first(s, """    float m[8], l[8], D[8], dq0[8], dq1[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        int tr = t0 + row0 + r;
        if (tr < seq) {
            int mi = tr * nh + h;
            m[r] = m_in[mi];
            l[r] = l_in[mi];
            float part = smDO[lane * QSTR + row0 + r] * O[qkv_at(tr, h, nh, dh) + lane]
                       + smDO[(lane + 32) * QSTR + row0 + r] * O[qkv_at(tr, h, nh, dh) + lane + 32];
            D[r] = warp_sum(part);
        } else {
            m[r] = NEG;
            l[r] = 0.f;
            D[r] = 0.f;
        }
        dq0[r] = 0.f;
        dq1[r] = 0.f;
    }""",
                   """    float m[8], l[8], D[8], dq0[8], dq1[8];
    float Dpart[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        int tr = t0 + row0 + r;
        Dpart[r] = 0.f;
        if (tr < seq) {
            int mi = tr * nh + h;
            m[r] = m_in[mi];
            l[r] = l_in[mi];
        } else {
            m[r] = NEG;
            l[r] = 0.f;
        }
        dq0[r] = 0.f;
        dq1[r] = 0.f;
    }
    // D = rowsum(dO·O)：按 head_dim 窗口累加（原来一次读 64 维）
#pragma unroll 1
    for (int dc = 0; dc < DH / DCHUNK; ++dc) {
        int d0 = dc * DCHUNK;
        load_win(dO, smDO, t0, h, nh, dh, tid, seq, d0, DCHUNK, QSTR);
        __syncthreads();
        if (lane < DCHUNK) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                int tr = t0 + row0 + r;
                if (tr < seq)
                    Dpart[r] += smDO[lane * QSTR + row0 + r] * O[qkv_at(tr, h, nh, dh) + d0 + lane];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        D[r] = (t0 + row0 + r < seq) ? warp_sum(Dpart[r]) : 0.f;
    }""", "D按窗口")

    s = rep_first(s, """        float sx[8], dp[8];
        flash_qk_dp(smQ, smDO, smK, smV, row0, lane, sx, dp);""",
                  """        float sx[8], dp[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) { sx[r] = 0.f; dp[r] = 0.f; }
#pragma unroll 1
        for (int dc = 0; dc < DH / DCHUNK; ++dc) {
            int d0 = dc * DCHUNK;
            load_win(Q, smQ, t0, h, nh, dh, tid, seq, d0, DCHUNK, QSTR);
            load_win(dO, smDO, t0, h, nh, dh, tid, seq, d0, DCHUNK, QSTR);
            __syncthreads();
            flash_qk_dp_win(smQ, smDO, smK, smV, row0, lane, d0, sx, dp);
            __syncthreads();
        }""", "主循环窗口")

    s = rep_first(s, """#pragma unroll 1
        for (int e = 0; e < 8; ++e) {
            int idx = tid + e * 256;
            int k = idx >> 6;
            int d = idx & 63;
            float ak = 0.f, av = 0.f;
#pragma unroll 4
            for (int q = 0; q < BT; ++q) {
                float s = smDS[q * BC + k];
                float p = smP[q * BC + k];
                ak = fmaf(s, smQ[d * QSTR + q], ak);
                av = fmaf(p, smDO[d * QSTR + q], av);
            }
            int tok = k0 + k;
            if (tok < seq) {
                float* dkrow = dK + qkv_at(tok, h, nh, dh);
                float* dvrow = dV + qkv_at(tok, h, nh, dh);
                atomicAdd(dkrow + d, scale * ak);
                atomicAdd(dvrow + d, av);
            }
        }""",
                   """#pragma unroll 1
        for (int dc = 0; dc < DH / DCHUNK; ++dc) {
            int d0 = dc * DCHUNK;
            load_win(Q, smQ, t0, h, nh, dh, tid, seq, d0, DCHUNK, QSTR);
            load_win(dO, smDO, t0, h, nh, dh, tid, seq, d0, DCHUNK, QSTR);
            __syncthreads();
            // 每窗口 BC*DCHUNK = 512 个 (key, dim) 对，256 线程 ⇒ 2 轮
#pragma unroll 1
            for (int e = 0; e < (BC * DCHUNK) / 256; ++e) {
                int idx = tid + e * 256;
                int k = idx / DCHUNK;
                int dl = idx - k * DCHUNK;
                float ak = 0.f, av = 0.f;
#pragma unroll 4
                for (int q = 0; q < BT; ++q) {
                    float s = smDS[q * BC + k];
                    float p = smP[q * BC + k];
                    ak = fmaf(s, smQ[dl * QSTR + q], ak);
                    av = fmaf(p, smDO[dl * QSTR + q], av);
                }
                int tok = k0 + k;
                if (tok < seq) {
                    float* dkrow = dK + qkv_at(tok, h, nh, dh) + d0 + dl;
                    float* dvrow = dV + qkv_at(tok, h, nh, dh) + d0 + dl;
                    atomicAdd(dkrow, scale * ak);
                    atomicAdd(dvrow, av);
                }
            }
            __syncthreads();
        }""", "dK/dV窗口")

    io.open(CU, "w", encoding="utf-8", newline="\n").write(s)
    print("\n全部替换完成（旧 flash_qk_dp 未动，g2 不受影响）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
