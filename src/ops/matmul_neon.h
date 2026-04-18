#pragma once
// matmul_neon.h — ARM NEON register micro-kernel for matrix multiplication.
//
// Included only by matmul.cpp.  All functions are static so they are not
// visible outside the translation unit.
//
// Requires B to be contiguous in the column direction (stride == 1).

#ifdef __ARM_NEON
#include <algorithm>
#include <arm_neon.h>
#include <omp.h>

// ─────────────────────────────────────────────────────────────────────────────
// neon_micro_kernel_4x16
//
// Computes a 4×16 tile of C += A[:,k] * B[k,:] entirely in registers.
//
// Each c{i}{j} variable is one float32x4_t register holding 4 consecutive
// output values.  The naming convention is c{row}{col_group}:
//
//         n+0..3   n+4..7   n+8..11  n+12..15
//   m+0:  c00      c01      c02      c03
//   m+1:  c10      c11      c12      c13
//   m+2:  c20      c21      c22      c23
//   m+3:  c30      c31      c32      c33
//
// The 16 accumulators are passed in and out by value so the compiler keeps
// them in physical NEON registers without spilling to the stack.
// ─────────────────────────────────────────────────────────────────────────────
struct NeonTile4x16 {
    float32x4_t c00, c01, c02, c03;
    float32x4_t c10, c11, c12, c13;
    float32x4_t c20, c21, c22, c23;
    float32x4_t c30, c31, c32, c33;
};

static inline NeonTile4x16 neon_load_tile(
        const float* c, size_t N, size_t m, size_t n) {
    return {
        vld1q_f32(c + (m+0)*N + n +  0), vld1q_f32(c + (m+0)*N + n +  4),
        vld1q_f32(c + (m+0)*N + n +  8), vld1q_f32(c + (m+0)*N + n + 12),
        vld1q_f32(c + (m+1)*N + n +  0), vld1q_f32(c + (m+1)*N + n +  4),
        vld1q_f32(c + (m+1)*N + n +  8), vld1q_f32(c + (m+1)*N + n + 12),
        vld1q_f32(c + (m+2)*N + n +  0), vld1q_f32(c + (m+2)*N + n +  4),
        vld1q_f32(c + (m+2)*N + n +  8), vld1q_f32(c + (m+2)*N + n + 12),
        vld1q_f32(c + (m+3)*N + n +  0), vld1q_f32(c + (m+3)*N + n +  4),
        vld1q_f32(c + (m+3)*N + n +  8), vld1q_f32(c + (m+3)*N + n + 12),
    };
}

static inline void neon_store_tile(
        float* c, size_t N, size_t m, size_t n, const NeonTile4x16& t) {
    vst1q_f32(c + (m+0)*N + n +  0, t.c00); vst1q_f32(c + (m+0)*N + n +  4, t.c01);
    vst1q_f32(c + (m+0)*N + n +  8, t.c02); vst1q_f32(c + (m+0)*N + n + 12, t.c03);
    vst1q_f32(c + (m+1)*N + n +  0, t.c10); vst1q_f32(c + (m+1)*N + n +  4, t.c11);
    vst1q_f32(c + (m+1)*N + n +  8, t.c12); vst1q_f32(c + (m+1)*N + n + 12, t.c13);
    vst1q_f32(c + (m+2)*N + n +  0, t.c20); vst1q_f32(c + (m+2)*N + n +  4, t.c21);
    vst1q_f32(c + (m+2)*N + n +  8, t.c22); vst1q_f32(c + (m+2)*N + n + 12, t.c23);
    vst1q_f32(c + (m+3)*N + n +  0, t.c30); vst1q_f32(c + (m+3)*N + n +  4, t.c31);
    vst1q_f32(c + (m+3)*N + n +  8, t.c32); vst1q_f32(c + (m+3)*N + n + 12, t.c33);
}

// Accumulate one k-step into the tile: for each of the 4 rows, broadcast
// A[m+i, k] and FMA it against the 4 B vectors covering columns n..n+15.
// B is loaded once and reused across all 4 rows — that's the register-blocking
// payoff: 4 loads of B serve 16 FMA instructions.
static inline void neon_accumulate_k(
        NeonTile4x16& t,
        const float* a, size_t sA0, size_t sA1,
        const float* b, size_t sB0,
        size_t m, size_t n, size_t k) {
    const float32x4_t b0 = vld1q_f32(b + k*sB0 + n +  0);
    const float32x4_t b1 = vld1q_f32(b + k*sB0 + n +  4);
    const float32x4_t b2 = vld1q_f32(b + k*sB0 + n +  8);
    const float32x4_t b3 = vld1q_f32(b + k*sB0 + n + 12);

    const float32x4_t a0 = vdupq_n_f32(a[(m+0)*sA0 + k*sA1]);
    t.c00 = vfmaq_f32(t.c00, a0, b0); t.c01 = vfmaq_f32(t.c01, a0, b1);
    t.c02 = vfmaq_f32(t.c02, a0, b2); t.c03 = vfmaq_f32(t.c03, a0, b3);

    const float32x4_t a1 = vdupq_n_f32(a[(m+1)*sA0 + k*sA1]);
    t.c10 = vfmaq_f32(t.c10, a1, b0); t.c11 = vfmaq_f32(t.c11, a1, b1);
    t.c12 = vfmaq_f32(t.c12, a1, b2); t.c13 = vfmaq_f32(t.c13, a1, b3);

    const float32x4_t a2 = vdupq_n_f32(a[(m+2)*sA0 + k*sA1]);
    t.c20 = vfmaq_f32(t.c20, a2, b0); t.c21 = vfmaq_f32(t.c21, a2, b1);
    t.c22 = vfmaq_f32(t.c22, a2, b2); t.c23 = vfmaq_f32(t.c23, a2, b3);

    const float32x4_t a3 = vdupq_n_f32(a[(m+3)*sA0 + k*sA1]);
    t.c30 = vfmaq_f32(t.c30, a3, b0); t.c31 = vfmaq_f32(t.c31, a3, b1);
    t.c32 = vfmaq_f32(t.c32, a3, b2); t.c33 = vfmaq_f32(t.c33, a3, b3);
}

// ─────────────────────────────────────────────────────────────────────────────
// neon_matmul_forward
//
// Full forward pass using the micro-kernel above.
//
// Two levels of tiling:
//   Outer (T=64):  OMP work distribution — (M/T)×(N/T) parallel items.
//   Inner (KB=256): k-blocking — keeps the B panel (KB×16×4 = 16 KB) and
//                   A strip (4×KB×4 = 4 KB) resident in L1 together.
//
// Rows/cols not divisible by MR/NR fall through to a scalar edge loop.
// ─────────────────────────────────────────────────────────────────────────────
static void neon_matmul_forward(
        float* c,
        const float* a, size_t sA0, size_t sA1,
        const float* b, size_t sB0,
        size_t M, size_t K, size_t N) {
    constexpr size_t MR = 4;
    constexpr size_t NR = 16;
    constexpr size_t T  = 64;
    constexpr size_t KB = 256;

    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t m0 = 0; m0 < M; m0 += T)
    for (size_t n0 = 0; n0 < N; n0 += T) {
        const size_t m_end  = std::min(m0 + T, M);
        const size_t n_end  = std::min(n0 + T, N);
        // Last m/n that starts a complete MR/NR-wide tile within this OMP block.
        const size_t m_full = m0 + (m_end - m0) / MR * MR;
        const size_t n_full = n0 + (n_end - n0) / NR * NR;

        // ── micro-kernel: full MR×NR tiles ───────────────────────────────
        for (size_t m = m0; m < m_full; m += MR)
        for (size_t n = n0; n < n_full; n += NR) {
            NeonTile4x16 tile = neon_load_tile(c, N, m, n);
            for (size_t k0i = 0; k0i < K; k0i += KB) {
                const size_t k_end = std::min(k0i + KB, K);
                for (size_t k = k0i; k < k_end; ++k)
                    neon_accumulate_k(tile, a, sA0, sA1, b, sB0, m, n, k);
            }
            neon_store_tile(c, N, m, n, tile);
        }

        // ── scalar edge: right strip (N not divisible by NR) ─────────────
        if (n_full < n_end) {
            for (size_t m = m0; m < m_end; ++m)
            for (size_t k = 0; k < K; ++k) {
                const float a_mk = a[m*sA0 + k*sA1];
                for (size_t n = n_full; n < n_end; ++n)
                    c[m*N + n] += a_mk * b[k*sB0 + n];
            }
        }

        // ── scalar edge: bottom strip (M not divisible by MR) ────────────
        // Only covers n0..n_full to avoid double-processing the corner.
        if (m_full < m_end) {
            for (size_t m = m_full; m < m_end; ++m)
            for (size_t k = 0; k < K; ++k) {
                const float a_mk = a[m*sA0 + k*sA1];
                for (size_t n = n0; n < n_full; ++n)
                    c[m*N + n] += a_mk * b[k*sB0 + n];
            }
        }
    }
}

#endif // __ARM_NEON
