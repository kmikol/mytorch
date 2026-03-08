#pragma once

#include <vector>
#include "tensorlib.h"


struct ReluOp {

    static Tensor forward(const Tensor& x) {
        int64_t rows = x.shape(0), cols = x.shape(1);
        Tensor out = Tensor::zeros({rows, cols});
        for (int64_t r = 0; r < rows; r++)
            for (int64_t c = 0; c < cols; c++)
                out.at({r,c}) = x.at({r,c}) > 0.f ? x.at({r,c}) : 0.f;
        return out;
    }

    // needs saved_input because we need to know which values were positive
    // the output alone doesn't tell us — a zero output could mean x was
    // exactly zero or negative, and the gradient rule is the same for both
    static std::vector<Tensor> backward(
        const Tensor& grad,
        const Tensor& saved_input)
    {
        int64_t rows = grad.shape(0), cols = grad.shape(1);
        Tensor dx = Tensor::zeros({rows, cols});
        for (int64_t r = 0; r < rows; r++)
            for (int64_t c = 0; c < cols; c++)
                // pass gradient through only where input was positive
                // everywhere else gradient is zero (ReLU was "off")
                dx.at({r,c}) = saved_input.at({r,c}) > 0.f ? grad.at({r,c}) : 0.f;
        return {dx};
    }
};

inline Tensor relu(const Tensor& x) {
    assert(x.ndim() == 2);

    // --- forward ---
    Tensor out = ReluOp::forward(x);

    // --- backward ---
    if (!x.requires_grad()) return out;

    Tensor saved_input = x.clone();
    out.autograd_meta = make_grad_meta(
        "relu",
        {x.autograd_meta},
        [saved_input](const Tensor& grad) {
            return ReluOp::backward(grad, saved_input);
        });

    return out;
}