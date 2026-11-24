// tgai_attn_tc.cu —— V100（sm_70 / Volta）的 fp16 张量核注意力前向。
//
// 为什么单独一个文件、单独一次 NVRTC 编译：
//   * mma/wmma 指令是**架构相关**的。主模块 `_device_source()` 会被编到
//     compute_60 / compute_61（P100、1060），那里没有张量核，混进去会让
//     整个主模块编不过 ⇒ 引擎在这些卡上直接不可用。这里按需编译、按架构拒绝。
//   * 默认路径（fp32 融合注意力 / 物化注意力）**一个字节都不改**：本内核只在
//     `TG_ATTN_TC=1` 且 q/k/v 都是 fp16 存储时才被调用（见 cuda.py::attn_tc_fwd）。
//
// 硬件事实（PTX ISA 文档 + 本机 V100 实测，见 lab/rent/probe_tc_isa*.py / env_tc3.txt）：
//   * `mma.sync.aligned.m16n8k8/k16` 的 .f16 形状要求 **sm_75+** —— V100 用不了。
//     网上/直觉里"V100 用 m16n8k8"是错的（FlashAttention 至今不支持 Volta 也是这个原因）。
//   * V100 能用的是：`wmma`（m16n16k16，sm_70+，编成 HMMA.884 序列）与
//     `mma.sync.aligned.m8n8k4`（sm_70+，一条指令 = 4 个独立 8x8x4，2048 FLOP）。
//   * 两者都必须在 `asm volatile` 里写 PTX；直接写 PTX 语句 NVRTC 不认。
//   * `wmma.load/store` 用**通用寻址**（不带 .shared）即可，实测数值精确。
//
// 算法（FlashAttention 式 online softmax，全部 fp32 累加/softmax）：
//   块网格 (ceil(seq/BQ), nh)，block 256 = 8 warp，每 warp 负责 16 行 query。
//   S = Q·K^T：wmma.m16n16k16（4 个 k 块 × 4 个 key 块），fp32 累加 → 存 smem
//   softmax：每线程一行的一半（32 列），warp 内两两 shuffle 归约，causal 掩码在此
//   P（fp16）→ smem → O = P·V：mma.m8n8k4（4 regs/线程的 A/B 片段，C 布局已知
//   ⇒ 可以直接按行乘 alpha 做 online rescale，这是不用 wmma 存 O 的原因）
//   O 全程 fp32 累加，最后一次除以 l 再舍入成 fp16 写回。
//
// smem 账（共 94208 B ≈ 92 KB，需要 cuFuncSetAttribute opt-in 到 V100 的 96 KB）：
//   smQ   [BQ=128][DH=64]  fp16  16384 B
//   smK   [BC=64][KS=80]   fp16  10240 B
//   smVT  [DH=64][KS=80]   fp16  10240 B  （V 转置，PV 的 B 操作数要列主序）
//   smS   [NW=8][16][SS=72] f32  36864 B  （每 warp 私有，S 落地做 softmax）
//   smP   [NW=8][16][KS=80] fp16 20480 B  （P 落地给 PV 的 A 操作数用）
// 行跨度 KS/SS 是 padding：64 half = 128 B = 恰好 32 个 bank，会让 16 行撞同一 bank；
// 80 half（160 B）/ 72 float（288 B）把冲突从 8~16 路降到 2~4 路（实测 17.3→24.6 TFLOPS），
// 同时满足 wmma 对 ldm 的对齐要求（f16 片段 32 B ⇒ 16 half 倍数；f32 ⇒ 8 float 倍数）。
//
// 契约：seq/nh 任意正整数；dh 恒为 64；输入是 fp16、连续、[B*H, seq, 64]
//       （head 步长 = seq*Dh、token 步长 = Dh，由 host 作为 tstride 传入）；
//       越界行/列补 0 并掩码。
// 局限（详见 docs/TC_ATTN_RESULTS.md）：只做前向；只支持 causal + dh==64。

#define BQ 128
#define BC 64
#define DH 64
#define NW 8
#define NT 256
/* smem 行跨度：加 padding 消 bank 冲突，且满足 wmma 的 ldm 对齐要求。
 * K/VT/P 是 f16（片段 32 B ⇒ ldm 必须是 16 half 倍数）：64 -> 80（160 B）。
 * S 是 f32（片段 32 B ⇒ ldm 必须是 8 float 倍数）：64 -> 72（288 B）。 */
#define KS 80
#define SS 72
#define NEG (-1.0e30f)

extern __shared__ __align__(32) unsigned short tc_smem[];

// ---------------------------------------------------------------- 基础指令
__device__ __forceinline__ unsigned short tc_f2h(float x) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    return h;
}

// 两个 fp32 → 一个 half2（低 16 位 = lo）
__device__ __forceinline__ unsigned int tc_pack2(float lo, float hi) {
    return (unsigned int)tc_f2h(lo) | ((unsigned int)tc_f2h(hi) << 16);
}

__device__ __forceinline__ void tc_wmma_ld_a_row(const void* p, unsigned int ldm,
                                                 unsigned int a[8]) {
    asm volatile(
        "wmma.load.a.sync.aligned.row.m16n16k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]),
          "=r"(a[4]), "=r"(a[5]), "=r"(a[6]), "=r"(a[7])
        : "l"(p), "r"(ldm));
}

__device__ __forceinline__ void tc_wmma_ld_b_col(const void* p, unsigned int ldm,
                                                 unsigned int b[8]) {
    asm volatile(
        "wmma.load.b.sync.aligned.col.m16n16k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;\n"
        : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]),
          "=r"(b[4]), "=r"(b[5]), "=r"(b[6]), "=r"(b[7])
        : "l"(p), "r"(ldm));
}

__device__ __forceinline__ void tc_wmma_store_d_row_f32(void* p, const float d[8],
                                                        unsigned int ldm) {
    asm volatile(
        "wmma.store.d.sync.aligned.row.m16n16k16.f32 [%0], "
        "{%1,%2,%3,%4,%5,%6,%7,%8}, %9;\n"
        :: "l"(p), "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]),
           "f"(d[4]), "f"(d[5]), "f"(d[6]), "f"(d[7]), "r"(ldm)
        : "memory");
}

// D = A(16x16, row) * B(16x16, col) + D
__device__ __forceinline__ void tc_wmma_mma(float d[8], const unsigned int a[8],
                                            const unsigned int b[8]) {
    asm volatile(
        "wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, "
        "{%8,%9,%10,%11,%12,%13,%14,%15}, "
        "{%16,%17,%18,%19,%20,%21,%22,%23}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));
}

// 4 个独立的 8x8x4：一条指令 = 2048 FLOP。
// 布局（PTX ISA 9.7.14.5.1，已在 V100 上逐元素验证）：
//   A 2 regs = {a0,a1,a2,a3}：row = lane%4 (+4 if lane>=16)，col = i（0..3）
//   B 2 regs：row = i，col = lane%4 (+4 if lane>=16)（列主序 → 同 row 的 k 连续）
//   C 8 regs：row = X (+4 if lane>=16), X = (lane&1)+(i&2)；col = (i&4)+(lane&2)+(i&1)
__device__ __forceinline__ void tc_mma_884(float acc[8], unsigned int a0, unsigned int a1,
                                           unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),
          "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------- 主内核
extern "C" __global__ void __launch_bounds__(NT)
attn_fwd_tc(const unsigned short* __restrict__ Q, const unsigned short* __restrict__ K,
            const unsigned short* __restrict__ V, unsigned short* __restrict__ O,
            float* __restrict__ M, float* __restrict__ LSE,
            int seq, int nh, int tstride, float scale) {
    const int h = (int)blockIdx.y;
    const int t0 = (int)blockIdx.x * BQ;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (t0 >= seq) return;

    unsigned short* smQ = tc_smem;
    unsigned short* smK = smQ + BQ * DH;
    unsigned short* smVT = smK + BC * KS;
    float* smS = (float*)(smVT + DH * KS);
    unsigned short* smP = (unsigned short*)(smS + NW * 16 * SS);
    float* swS = smS + warp * 16 * SS;
    unsigned short* swP = smP + warp * 16 * KS;

    // 每线程在 PV 阶段负责的 O 元素位置：
    //   j = 所属 comp（0..3）→ 行块 rbO=j>>1、dh 列块 dhalf=j&1
    const int j = (lane < 16) ? (lane >> 2) : ((lane - 16) >> 2);
    const int rbO = j >> 1;
    const int dhalf = j & 1;
    const int hi = (lane < 16) ? 0 : 4;
    const int orowA = rbO * 8 + (lane & 1) + hi;   // O 的（warp 内）第一行
    const int orowB = orowA + 2;                   // 第二行
    const int arow = rbO * 8 + (lane & 3) + hi;    // PV 的 A 片段行
    // 软最大阶段：每线程 = 一行（lane>>1）的一半（32 列）
    const int srow = lane >> 1;
    const int cbase = (lane & 1) * 32;
    const int qrow = t0 + warp * 16 + srow;

    // ---- 载入 Q 块（越界行补 0）----
    {
        const int n8 = BQ * (DH / 8);
        for (int i = tid; i < n8; i += NT) {
            const int r = i >> 3;
            const int c8 = (i & 7) << 3;
            const int tok = t0 + r;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (tok < seq)
                v = *reinterpret_cast<const uint4*>(Q + ((size_t)h * seq + tok) * tstride + c8);
            *reinterpret_cast<uint4*>(smQ + r * DH + c8) = v;
        }
    }
    // ---- 第一个 K/V 块（V 同时转置）----
    {
        const int n8 = BC * (DH / 8);
        for (int i = tid; i < n8; i += NT) {
            const int r = i >> 3;
            const int c8 = (i & 7) << 3;
            const int tok = r;                       // k0 = 0
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            uint4 w = make_uint4(0u, 0u, 0u, 0u);
            if (tok < seq) {
                v = *reinterpret_cast<const uint4*>(K + ((size_t)h * seq + tok) * tstride + c8);
                w = *reinterpret_cast<const uint4*>(V + ((size_t)h * seq + tok) * tstride + c8);
            }
            *reinterpret_cast<uint4*>(smK + r * KS + c8) = v;
            unsigned int u[4];
            u[0] = w.x; u[1] = w.y; u[2] = w.z; u[3] = w.w;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                smVT[(c8 + 2 * q) * KS + r] = (unsigned short)(u[q] & 0xffffu);
                smVT[(c8 + 2 * q + 1) * KS + r] = (unsigned short)(u[q] >> 16);
            }
        }
    }
    __syncthreads();

    // ---- Q 的 A 片段：整块内不变，只装一次进寄存器（4 个 k 块 × 8 regs）----
    unsigned int qf[4][8];
    {
        const int rb0 = warp * 16;
#pragma unroll
        for (int d = 0; d < 4; ++d)
            tc_wmma_ld_a_row(smQ + rb0 * DH + d * 16, DH, qf[d]);
    }

    const int k_end = (t0 + BQ < seq) ? (t0 + BQ) : seq;
    float mrow = NEG;
    float lrow = 0.f;
    float of[4][8];
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int i = 0; i < 8; ++i) of[c][i] = 0.f;

#pragma unroll 1
    for (int k0 = 0; k0 < k_end; k0 += BC) {
        // ---- S = Q·K^T（16 行 × 64 key，fp32 累加）----
        float sf[4][8];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt)
#pragma unroll
            for (int i = 0; i < 8; ++i) sf[kt][i] = 0.f;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                unsigned int bf[8];
                tc_wmma_ld_b_col(smK + (kt * 16) * KS + d * 16, KS, bf);
                tc_wmma_mma(sf[kt], qf[d], bf);
            }
        }
#pragma unroll
        for (int kt = 0; kt < 4; ++kt)
            tc_wmma_store_d_row_f32(swS + kt * 16, sf[kt], SS);
        __syncwarp();

        // ---- softmax（fp32，含 causal 掩码）----
        float s[32];
        float mx = NEG;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int col = k0 + cbase + i;
            const float v = swS[srow * SS + cbase + i] * scale;
            if (col <= qrow && col < seq) {
                s[i] = v;
                mx = fmaxf(mx, v);
            } else {
                s[i] = NEG;
            }
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        const float mnew = fmaxf(mrow, mx);
        const float alpha = (mrow > -1.0e20f) ? __expf(mrow - mnew) : 0.f;
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const float p = (s[i] > -1.0e20f) ? __expf(s[i] - mnew) : 0.f;
            s[i] = p;
            sum += p;
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        lrow = lrow * alpha + sum;
        mrow = mnew;
        // P 写成 fp16 落到 smem（行主序 [16][BC]），给 PV 的 A 操作数读
#pragma unroll
        for (int i = 0; i < 32; i += 8) {
            uint4 u;
            u.x = tc_pack2(s[i], s[i + 1]);
            u.y = tc_pack2(s[i + 2], s[i + 3]);
            u.z = tc_pack2(s[i + 4], s[i + 5]);
            u.w = tc_pack2(s[i + 6], s[i + 7]);
            *reinterpret_cast<uint4*>(swP + srow * KS + cbase + i) = u;
        }

        // ---- O = alpha·O + P·V（fp32 累加；按行做 online rescale）----
        const float aA = __shfl_sync(0xffffffffu, alpha, 2 * orowA);
        const float aB = __shfl_sync(0xffffffffu, alpha, 2 * orowB);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            of[c][0] *= aA; of[c][1] *= aA; of[c][4] *= aA; of[c][5] *= aA;
            of[c][2] *= aB; of[c][3] *= aB; of[c][6] *= aB; of[c][7] *= aB;
        }
        __syncwarp();
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
            const unsigned short* ap = swP + arow * KS + ks * 4;
            const unsigned int a0 = *reinterpret_cast<const unsigned int*>(ap);
            const unsigned int a1 = *reinterpret_cast<const unsigned int*>(ap + 2);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int dh = 16 * c + 8 * dhalf + (lane & 3) + hi;
                const unsigned short* bp = smVT + dh * KS + ks * 4;
                const unsigned int b0 = *reinterpret_cast<const unsigned int*>(bp);
                const unsigned int b1 = *reinterpret_cast<const unsigned int*>(bp + 2);
                tc_mma_884(of[c], a0, a1, b0, b1);
            }
        }

        __syncthreads();
        // ---- 预取下一块 K/V ----
        const int next = k0 + BC;
        if (next < k_end) {
            const int n8 = BC * (DH / 8);
            for (int i = tid; i < n8; i += NT) {
                const int r = i >> 3;
                const int c8 = (i & 7) << 3;
                const int tok = next + r;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                uint4 w = make_uint4(0u, 0u, 0u, 0u);
                if (tok < seq) {
                    v = *reinterpret_cast<const uint4*>(K + ((size_t)h * seq + tok) * tstride + c8);
                    w = *reinterpret_cast<const uint4*>(V + ((size_t)h * seq + tok) * tstride + c8);
                }
                *reinterpret_cast<uint4*>(smK + r * KS + c8) = v;
                unsigned int u[4];
                u[0] = w.x; u[1] = w.y; u[2] = w.z; u[3] = w.w;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    smVT[(c8 + 2 * q) * KS + r] = (unsigned short)(u[q] & 0xffffu);
                    smVT[(c8 + 2 * q + 1) * KS + r] = (unsigned short)(u[q] >> 16);
                }
            }
            __syncthreads();
        }
    }

    // ---- 写回 O（除以 l 后舍入 fp16）/ M / LSE ----
    const float lA = __shfl_sync(0xffffffffu, lrow, 2 * orowA);
    const float lB = __shfl_sync(0xffffffffu, lrow, 2 * orowB);
    const float invA = (lA > 0.f) ? (1.f / lA) : 0.f;
    const float invB = (lB > 0.f) ? (1.f / lB) : 0.f;
    const int qrA = t0 + warp * 16 + orowA;
    const int qrB = t0 + warp * 16 + orowB;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int dhA = 16 * c + 8 * dhalf + (lane & 2);
        if (qrA < seq) {
            unsigned short* op = O + ((size_t)h * seq + qrA) * tstride + dhA;
            *reinterpret_cast<unsigned int*>(op) =
                tc_pack2(of[c][0] * invA, of[c][1] * invA);
            *reinterpret_cast<unsigned int*>(op + 4) =
                tc_pack2(of[c][4] * invA, of[c][5] * invA);
        }
        if (qrB < seq) {
            unsigned short* op = O + ((size_t)h * seq + qrB) * tstride + dhA;
            *reinterpret_cast<unsigned int*>(op) =
                tc_pack2(of[c][2] * invB, of[c][3] * invB);
            *reinterpret_cast<unsigned int*>(op + 4) =
                tc_pack2(of[c][6] * invB, of[c][7] * invB);
        }
    }
    if (qrow < seq && (lane & 1) == 0) {
        const size_t idx = (size_t)h * seq + qrow;
        M[idx] = mrow;
        LSE[idx] = mrow + __logf(lrow);
    }
}
