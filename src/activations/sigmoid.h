#pragma once
#include <vector>
#include <cmath>
#include "tensorlib.h"

struct SigmoidOp {

    static Tensor forward(const Tensor& x) {
        int64_t rows = x.shape(0), cols = x.shape(1);
        Tensor out = Tensor::zeros({rows, cols});
        for (int64_t r = 0; r < rows; r++)
            for (int64_t c = 0; c < cols; c++)
                out.at({r,c}) = 1.f / (1.f + std::exp(-x.at({r,c})));
        return out;
    }

    // needs saved_out because the derivative is: out * (1 - out)
    // we don't need the original input — just what came out of sigmoid
    static std::vector<Tensor> backward(
        const Tensor& grad,
        const Tensor& saved_out)
    {
        int64_t rows = grad.shape(0), cols = grad.shape(1);
        Tensor dx = Tensor::zeros({rows, cols});
        for (int64_t r = 0; r < rows; r++)
            for (int64_t c = 0; c < cols; c++) {
                float s = saved_out.at({r,c});
                dx.at({r,c}) = grad.at({r,c}) * s * (1.f - s);
            }
        return {dx};
    }
};

inline Tensor sigmoid(const Tensor& x) {
    assert(x.ndim() == 2);

    // --- forward ---
    Tensor out = SigmoidOp::forward(x);

    // --- backward ---
    if (!x.requires_grad()) return out;

    Tensor saved_out = out.clone();
    out.autograd_meta = make_grad_meta(
        "sigmoid",
        {x.autograd_meta},
        [saved_out](const Tensor& grad) {
            return SigmoidOp::backward(grad, saved_out);
        });

    return out;
}