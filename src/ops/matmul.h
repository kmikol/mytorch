#pragma once
#include <vector>
#include "tensorlib.h"

inline Tensor matmul(const Tensor& A, const Tensor& B);

struct MatMulOp {

    // called once, saves what backward needs
    static Tensor forward(const Tensor& A, const Tensor& B) {

        int64_t M = A.shape(0), K = A.shape(1), N = B.shape(1);
        Tensor C = Tensor::zeros({M, N});

        // fast path for contiguous tensors — can use raw pointer arithmetic
        if (A.is_contiguous() && B.is_contiguous() && C.is_contiguous()) {
            const float* a = A.implementation->storage->ptr();
            const float* b = B.implementation->storage->ptr();
            float*       c = C.implementation->storage->ptr();

            for (int64_t m = 0; m < M; m++)
                for (int64_t k = 0; k < K; k++)
                    for (int64_t n = 0; n < N; n++)
                        c[m*N + n] += a[m*K + k] * b[k*N + n];

        } else {
            // general path for non-contiguous tensors
            // Access elements using at() which handles strides correctly, but is slower
            for (int64_t m = 0; m < M; m++)
                for (int64_t k = 0; k < K; k++)
                    for (int64_t n = 0; n < N; n++)
                        C.at({m,n}) += A.at({m,k}) * B.at({k,n});
        }

        return C;
    }

    // called during backward — receives upstream gradient
    static std::vector<Tensor> backward(
        const Tensor& grad,
        const Tensor& saved_A, const Tensor& saved_B,
        bool rA, bool rB)
    {
        std::vector<Tensor> grads(2);
        if (rA) grads[0] = matmul(grad, saved_B.transpose());
        if (rB) grads[1] = matmul(saved_A.transpose(), grad);
        return grads;
    }
};

inline Tensor matmul(const Tensor& A, const Tensor& B) {
    assert(A.ndim() == 2 && B.ndim() == 2);
    assert(A.shape(1) == B.shape(0));

    Tensor C = MatMulOp::forward(A, B);

    bool rA = A.requires_grad(), rB = B.requires_grad();
    if (grad_mode_enabled && (rA || rB)) {
        
        NoGradGuard no_grad;
        
        Tensor sA = A.clone(), sB = B.clone();
        auto mA = A.autograd_meta, mB = B.autograd_meta;

        C.autograd_meta = make_grad_meta("matmul", {mA, mB},
            [sA, sB, rA, rB](const Tensor& grad) {
                return MatMulOp::backward(grad, sA, sB, rA, rB);
            });
    }
    return C;
}