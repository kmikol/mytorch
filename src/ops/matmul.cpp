#include "ops/matmul.h"

#include <algorithm>
#include <cassert>
#include <omp.h>


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
