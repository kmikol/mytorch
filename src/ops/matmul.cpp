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
    Tensor C = Tensor::zeros(out_shape, 2);  // zero-init so += accumulation is correct

    float*       c   = C.storage->data;                 // contiguous, offset = 0
    const float* a   = A.storage->data + A.offset;
    const float* b   = B.storage->data + B.offset;

    // Cache strides once — avoids repeated indirection in the inner loop.
    // Contiguous A: sA0 = K, sA1 = 1.  Transposed A: sA0 = 1, sA1 = K.
    size_t sA0 = A.strides[0], sA1 = A.strides[1];
    size_t sB0 = B.strides[0], sB1 = B.strides[1];

#ifdef __ARM_NEON
    // ── NEON register micro-kernel (B contiguous in n, sB1 == 1) ──────────
    //
    // Tile: MR=4 rows × NR=16 cols.  The 4×4 = 16 accumulator float32x4_t
    // registers stay live for the entire K loop — C is loaded once and stored
    // once per output tile, eliminating the load-add-store on every k step
    // that the scalar path does.
    //
    // K is blocked in steps of KB to keep the B panel (KB×16 floats = 16 KB
    // at KB=256) resident in L1.  A strip (4×KB = 4 KB) fits alongside it.
    //
    // OMP parallelises over (m0, n0) cache tiles of size T×T.
    if (sB1 == 1) {
        constexpr size_t MR = 4;
        constexpr size_t NR = 16;
        constexpr size_t T  = 64;   // OMP tile — small enough for parallelism at any N
        constexpr size_t KB = 256;  // k-blocking inside micro-kernel (L1 reuse of B)

        #pragma omp parallel for schedule(static) collapse(2)
        for (size_t m0 = 0; m0 < M; m0 += T)
        for (size_t n0 = 0; n0 < N; n0 += T) {
            const size_t m_end  = std::min(m0 + T, M);
            const size_t n_end  = std::min(n0 + T, N);
            const size_t m_full = m0 + (m_end - m0) / MR * MR;
            const size_t n_full = n0 + (n_end - n0) / NR * NR;

            // ── full MR×NR tiles ─────────────────────────────────────────
            for (size_t m = m0; m < m_full; m += MR)
            for (size_t n = n0; n < n_full; n += NR) {

                // Load C tile into 16 accumulator registers (one load each).
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

                // Accumulate over all K, blocked by KB for L1 reuse of B.
                for (size_t k0i = 0; k0i < K; k0i += KB) {
                    const size_t k_end = std::min(k0i + KB, K);
                    for (size_t k = k0i; k < k_end; ++k) {
                        const float32x4_t b0 = vld1q_f32(b + k*sB0 + n +  0);
                        const float32x4_t b1 = vld1q_f32(b + k*sB0 + n +  4);
                        const float32x4_t b2 = vld1q_f32(b + k*sB0 + n +  8);
                        const float32x4_t b3 = vld1q_f32(b + k*sB0 + n + 12);

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

                // Store C tile back (one store each).
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

            // ── scalar edge: right strip (N not divisible by NR) ─────────
            if (n_full < n_end) {
                for (size_t m = m0; m < m_end; ++m)
                for (size_t k = 0; k < K; ++k) {
                    const float a_mk = a[m*sA0 + k*sA1];
                    for (size_t n = n_full; n < n_end; ++n)
                        c[m*N + n] += a_mk * b[k*sB0 + n];
                }
            }

            // ── scalar edge: bottom strip (M not divisible by MR) ────────
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

    // ── Scalar fallback — handles non-contiguous B (e.g. transposed inputs) ──
    constexpr size_t T = 64;

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
