#include "ops/matmul.h"

#include <algorithm>
#include <cassert>
#include <omp.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif


Tensor MatMulOp::forward(const Tensor& A, const Tensor& B) {
    assert(A.ndim == 2 && B.ndim == 2);
    assert(A.shape[1] == B.shape[0]);

    size_t M = A.shape[0], K = A.shape[1], N = B.shape[1];

    Shape out_shape{};
    out_shape[0] = M;
    out_shape[1] = N;
    // Zero-initialised so the += accumulation in every path is correct from
    // the first step without a separate zeroing pass.
    Tensor C = Tensor::zeros(out_shape, 2);

    float*       c = C.storage->data;
    const float* a = A.storage->data + A.offset;
    const float* b = B.storage->data + B.offset;

    // Read strides once into locals.  A tensor produced by .T() has its two
    // stride values swapped but the underlying data is unchanged, so using
    // strides here lets the same kernel handle both normal and transposed
    // inputs without any data copy.
    //   Contiguous [M×K]:  sA0=K, sA1=1  → A[m,k] = a[m*K + k]
    //   Transposed [K×M]:  sA0=1, sA1=K  → A[m,k] = a[m*1 + k*K]  (same memory)
    size_t sA0 = A.strides[0], sA1 = A.strides[1];
    size_t sB0 = B.strides[0], sB1 = B.strides[1];

#ifdef __ARM_NEON
    // ── NEON register micro-kernel ────────────────────────────────────────
    //
    // Only entered when B is contiguous in the column direction (sB1 == 1),
    // which is true for any normal (non-transposed) matrix.  The backward
    // pass always passes a transposed B so it falls through to the scalar
    // path below.
    //
    // Core idea — eliminate the C memory round-trip on every k step:
    //   Scalar path does per k:  load C[m,n]  →  add  →  store C[m,n]
    //   Micro-kernel does:       load C tile ONCE → K×FMAs in registers → store ONCE
    //
    // Output tile size: MR=4 rows × NR=16 cols = 64 floats.
    // That requires 16 float32x4_t accumulator registers (c00..c33).
    // ARM has 32 NEON registers total, leaving 16 for A broadcasts and B loads.
    if (sB1 == 1) {
        constexpr size_t MR = 4;    // rows processed per micro-kernel call
        constexpr size_t NR = 16;   // cols processed per micro-kernel call (4 NEON regs × 4 floats)
        constexpr size_t T  = 64;   // OMP tile size — (N/T)² work items, must be ≥ parallelism needed
        constexpr size_t KB = 256;  // k-strip size — keeps B panel (KB×NR×4 = 16 KB) in L1

        // Parallelise over (m0, n0) output tiles.  collapse(2) flattens the
        // two loops into one index space so OMP sees (M/T)×(N/T) independent
        // items and can distribute them evenly.  schedule(static) pre-assigns
        // items to threads without a work queue — valid because every tile
        // covers exactly T×T output elements (uniform work).
        #pragma omp parallel for schedule(static) collapse(2)
        for (size_t m0 = 0; m0 < M; m0 += T)
        for (size_t n0 = 0; n0 < N; n0 += T) {
            const size_t m_end  = std::min(m0 + T, M);
            const size_t n_end  = std::min(n0 + T, N);
            // Last row/col that starts a full MR/NR-wide tile.
            // Rows m_full..m_end and cols n_full..n_end are handled by the
            // scalar edge loops below.
            const size_t m_full = m0 + (m_end - m0) / MR * MR;
            const size_t n_full = n0 + (n_end - n0) / NR * NR;

            // ── full MR×NR micro-kernel tiles ─────────────────────────────
            for (size_t m = m0; m < m_full; m += MR)
            for (size_t n = n0; n < n_full; n += NR) {

                // Load the 4×16 output tile from C into 16 NEON registers.
                // Naming: c{row}{col_group}, col_group 0..3 covers n+[0..3],
                // n+[4..7], n+[8..11], n+[12..15] respectively.
                // This is the ONLY time C is read for this tile; all K
                // accumulation happens in these registers without touching memory.
                float32x4_t c00 = vld1q_f32(c + (m+0)*N + n +  0);
                float32x4_t c01 = vld1q_f32(c + (m+0)*N + n +  4);
                float32x4_t c02 = vld1q_f32(c + (m+0)*N + n +  8);
                float32x4_t c03 = vld1q_f32(c + (m+0)*N + n + 12);
                float32x4_t c10 = vld1q_f32(c + (m+1)*N + n +  0);
                float32x4_t c11 = vld1q_f32(c + (m+1)*N + n +  4);
                float32x4_t c12 = vld1q_f32(c + (m+1)*N + n +  8);
                float32x4_t c13 = vld1q_f32(c + (m+1)*N + n + 12);
                float32x4_t c20 = vld1q_f32(c + (m+2)*N + n +  0);
                float32x4_t c21 = vld1q_f32(c + (m+2)*N + n +  4);
                float32x4_t c22 = vld1q_f32(c + (m+2)*N + n +  8);
                float32x4_t c23 = vld1q_f32(c + (m+2)*N + n + 12);
                float32x4_t c30 = vld1q_f32(c + (m+3)*N + n +  0);
                float32x4_t c31 = vld1q_f32(c + (m+3)*N + n +  4);
                float32x4_t c32 = vld1q_f32(c + (m+3)*N + n +  8);
                float32x4_t c33 = vld1q_f32(c + (m+3)*N + n + 12);

                // Accumulate over all K, blocked in strips of KB.
                // Without blocking the B panel per micro-kernel call would be
                // K×NR×4 bytes (64 KB at K=1024), evicting A from L1 mid-loop.
                // With KB=256 the B strip is 16 KB and the A strip is 4 KB —
                // both fit in L1 together.  The accumulators above stay live
                // across all k0i iterations; they are never spilled.
                for (size_t k0i = 0; k0i < K; k0i += KB) {
                    const size_t k_end = std::min(k0i + KB, K);
                    for (size_t k = k0i; k < k_end; ++k) {
                        // Load 16 consecutive B values from row k.
                        // sB1==1 guarantees these are contiguous in memory, so
                        // each pair of loads shares one 64-byte cache line.
                        const float32x4_t b0 = vld1q_f32(b + k*sB0 + n +  0);
                        const float32x4_t b1 = vld1q_f32(b + k*sB0 + n +  4);
                        const float32x4_t b2 = vld1q_f32(b + k*sB0 + n +  8);
                        const float32x4_t b3 = vld1q_f32(b + k*sB0 + n + 12);

                        // Broadcast A[m+i, k] into all 4 lanes of a NEON register,
                        // then FMA against the four B vectors.
                        // All four output rows reuse the same b0..b3 — loading B
                        // once for MR=4 rows is the register-blocking payoff.
                        // vfmaq_f32(acc, a, b) computes acc + a*b for all 4 lanes.
                        const float32x4_t a0 = vdupq_n_f32(a[(m+0)*sA0 + k*sA1]);
                        c00 = vfmaq_f32(c00, a0, b0);
                        c01 = vfmaq_f32(c01, a0, b1);
                        c02 = vfmaq_f32(c02, a0, b2);
                        c03 = vfmaq_f32(c03, a0, b3);

                        const float32x4_t a1 = vdupq_n_f32(a[(m+1)*sA0 + k*sA1]);
                        c10 = vfmaq_f32(c10, a1, b0);
                        c11 = vfmaq_f32(c11, a1, b1);
                        c12 = vfmaq_f32(c12, a1, b2);
                        c13 = vfmaq_f32(c13, a1, b3);

                        const float32x4_t a2 = vdupq_n_f32(a[(m+2)*sA0 + k*sA1]);
                        c20 = vfmaq_f32(c20, a2, b0);
                        c21 = vfmaq_f32(c21, a2, b1);
                        c22 = vfmaq_f32(c22, a2, b2);
                        c23 = vfmaq_f32(c23, a2, b3);

                        const float32x4_t a3 = vdupq_n_f32(a[(m+3)*sA0 + k*sA1]);
                        c30 = vfmaq_f32(c30, a3, b0);
                        c31 = vfmaq_f32(c31, a3, b1);
                        c32 = vfmaq_f32(c32, a3, b2);
                        c33 = vfmaq_f32(c33, a3, b3);
                    }
                }

                // Write the fully-accumulated tile back to C.
                // This is the ONLY write for this 4×16 region across all K steps.
                vst1q_f32(c + (m+0)*N + n +  0, c00);
                vst1q_f32(c + (m+0)*N + n +  4, c01);
                vst1q_f32(c + (m+0)*N + n +  8, c02);
                vst1q_f32(c + (m+0)*N + n + 12, c03);
                vst1q_f32(c + (m+1)*N + n +  0, c10);
                vst1q_f32(c + (m+1)*N + n +  4, c11);
                vst1q_f32(c + (m+1)*N + n +  8, c12);
                vst1q_f32(c + (m+1)*N + n + 12, c13);
                vst1q_f32(c + (m+2)*N + n +  0, c20);
                vst1q_f32(c + (m+2)*N + n +  4, c21);
                vst1q_f32(c + (m+2)*N + n +  8, c22);
                vst1q_f32(c + (m+2)*N + n + 12, c23);
                vst1q_f32(c + (m+3)*N + n +  0, c30);
                vst1q_f32(c + (m+3)*N + n +  4, c31);
                vst1q_f32(c + (m+3)*N + n +  8, c32);
                vst1q_f32(c + (m+3)*N + n + 12, c33);
            }

            // ── scalar edge: right strip (N not divisible by NR=16) ───────
            // Covers columns n_full..n_end for every row in this OMP tile.
            if (n_full < n_end) {
                for (size_t m = m0; m < m_end; ++m)
                for (size_t k = 0; k < K; ++k) {
                    const float a_mk = a[m*sA0 + k*sA1];
                    for (size_t n = n_full; n < n_end; ++n)
                        c[m*N + n] += a_mk * b[k*sB0 + n];
                }
            }

            // ── scalar edge: bottom strip (M not divisible by MR=4) ───────
            // Covers rows m_full..m_end, but only columns n0..n_full to avoid
            // double-processing the bottom-right corner (already handled above).
            if (m_full < m_end) {
                for (size_t m = m_full; m < m_end; ++m)
                for (size_t k = 0; k < K; ++k) {
                    const float a_mk = a[m*sA0 + k*sA1];
                    for (size_t n = n0; n < n_full; ++n)
                        c[m*N + n] += a_mk * b[k*sB0 + n];
                }
            }
        }
        return C;
    }
#endif

    // ── Scalar fallback ───────────────────────────────────────────────────
    //
    // Used when B is non-contiguous in the column direction (sB1 != 1), which
    // happens whenever B is transposed — i.e. on every backward pass.
    //
    // Three-level cache tiling: the working set per iteration is three T×T
    // tiles ≈ 3×64×64×4 = 48 KB, which fits in L1 and avoids evicting A or B
    // before all their values have been reused.
    //
    // Inner loop order is m-k-n: hoisting a[m,k] outside the n loop saves one
    // load per (m,k) pair instead of one per (m,k,n) triple.
    constexpr size_t T = 64;

    // schedule(dynamic) because boundary tiles can be smaller than T×T,
    // making work per item uneven — dynamic assignment keeps threads busy.
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (size_t m0 = 0; m0 < M; m0 += T)
    for (size_t n0 = 0; n0 < N; n0 += T)
    for (size_t k0 = 0; k0 < K; k0 += T) {

        size_t m_end = std::min(m0 + T, M);
        size_t n_end = std::min(n0 + T, N);
        size_t k_end = std::min(k0 + T, K);

        for (size_t m = m0; m < m_end; ++m)
        for (size_t k = k0; k < k_end; ++k) {
            float a_mk = a[m * sA0 + k * sA1];  // hoisted — one load per (m,k)
            for (size_t n = n0; n < n_end; ++n)
                c[m * N + n] += a_mk * b[k * sB0 + n * sB1];
        }
    }
    return C;
}

std::vector<Tensor> MatMulOp::backward(const Tensor& grad,
                                       const Tensor& A,
                                       const Tensor& B) {
    // dL/dA = grad @ B^T  — forward() handles the transposed strides natively
    // dL/dB = A^T  @ grad
    return {
        MatMulOp::forward(grad,   B.T()),
        MatMulOp::forward(A.T(), grad),
    };
}

Tensor matmul(const Tensor& A, const Tensor& B) {
    assert(A.ndim == 2 && B.ndim == 2);
    assert(A.shape[1] == B.shape[0]);

    Tensor C = MatMulOp::forward(A, B);

    bool rA = A.requires_grad();
    bool rB = B.requires_grad();

    if (grad_mode_enabled && (rA || rB)) {
        std::vector<std::shared_ptr<AutogradMeta>> active_metas;
        if (rA) active_metas.push_back(A.autograd_meta);
        if (rB) active_metas.push_back(B.autograd_meta);

        C.autograd_meta = make_grad_meta(
            "matmul",
            std::move(active_metas),
            [a_save = A, b_save = B, rA, rB](const Tensor& grad) {
                std::vector<Tensor> grads;
                if (rA) grads.push_back(MatMulOp::forward(grad,        b_save.T()));
                if (rB) grads.push_back(MatMulOp::forward(a_save.T(),  grad));
                return grads;
            }
        );
    }

    return C;
}
