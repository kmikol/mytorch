#pragma once

#include <vector>
#include "tensorlib.h"

struct MulOp {

    static Tensor forward(const Tensor& A, const Tensor& B) {
        int64_t rows = A.shape(0), cols = A.shape(1);
        Tensor C = Tensor::zeros({rows, cols});

        if (A.is_contiguous() && B.is_contiguous() && C.is_contiguous()) {
            const float* a = A.implementation->storage->ptr();
            const float* b = B.implementation->storage->ptr();
            float*       c = C.implementation->storage->ptr();

            int64_t n = rows * cols;
            for (int64_t i = 0; i < n; ++i)
                c[i] = a[i] * b[i];
        } else {
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t c = 0; c < cols; ++c)
                    C.at({r,c}) = A.at({r,c}) * B.at({r,c});
        }

        return C;
    }

    static std::vector<Tensor> backward(
        const Tensor& grad,
        const Tensor& saved_A,
        const Tensor& saved_B,
        bool rA, bool rB)
    {
        int64_t rows = grad.shape(0), cols = grad.shape(1);

        std::vector<Tensor> grads(2);

        if (rA) {
            Tensor gA = Tensor::zeros({rows, cols});
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t c = 0; c < cols; ++c)
                    gA.at({r,c}) = grad.at({r,c}) * saved_B.at({r,c});
            grads[0] = gA;
        }

        if (rB) {
            Tensor gB = Tensor::zeros({rows, cols});
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t c = 0; c < cols; ++c)
                    gB.at({r,c}) = grad.at({r,c}) * saved_A.at({r,c});
            grads[1] = gB;
        }

        return grads;
    }
};

inline Tensor mul(const Tensor& A, const Tensor& B) {
    assert(A.ndim() == 2 && B.ndim() == 2);
    assert(A.shape(0) == B.shape(0) && A.shape(1) == B.shape(1));

    Tensor C = MulOp::forward(A, B);

    bool rA = A.requires_grad(), rB = B.requires_grad();
    if (!grad_mode_enabled || !(rA || rB)) return C;

    NoGradGuard no_grad;

    Tensor sA = A.clone(), sB = B.clone();
    auto mA = A.autograd_meta, mB = B.autograd_meta;

    C.autograd_meta = make_grad_meta(
        "mul",
        {mA, mB},
        [sA, sB, rA, rB](const Tensor& grad) {
            return MulOp::backward(grad, sA, sB, rA, rB);
        });

    return C;
}