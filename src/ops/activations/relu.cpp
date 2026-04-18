#include "ops/activations/relu.h"

#include <cassert>


// Stride-aware element read: maps logical flat index i to the storage value in t.
static float elem_at(const Tensor& t, size_t i, const Strides& cs) {
    size_t idx = t.offset, rem = i;
    for (size_t d = 0; d < t.ndim; ++d) {
        size_t di  = rem / cs[d];
        rem       %= cs[d];
        idx       += di * t.strides[d];
    }
    return t.storage->data[idx];
}


Tensor ReLUOp::forward(const Tensor& x) {
    Tensor out(x.shape, x.ndim);
    float* o = out.storage->data;  // out is contiguous, offset = 0

    if (x.is_contiguous()) {
        const float* xp = x.storage->data + x.offset;
        for (size_t i = 0; i < x.numel; ++i)
            o[i] = xp[i] > 0.f ? xp[i] : 0.f;
    } else {
        Strides cs = Tensor::strides_from_shape(x.shape, x.ndim);
        for (size_t i = 0; i < x.numel; ++i) {
            float v = elem_at(x, i, cs);
            o[i] = v > 0.f ? v : 0.f;
        }
    }

    return out;
}


Tensor ReLUOp::backward(const Tensor& grad, const Tensor& x) {
    assert(grad.is_contiguous());
    Tensor grad_x(x.shape, x.ndim);
    const float* g  = grad.storage->data;
    float*       gx = grad_x.storage->data;

    if (x.is_contiguous()) {
        const float* xp = x.storage->data + x.offset;
        for (size_t i = 0; i < x.numel; ++i)
            gx[i] = g[i] * (xp[i] > 0.f ? 1.f : 0.f);
    } else {
        Strides cs = Tensor::strides_from_shape(x.shape, x.ndim);
        for (size_t i = 0; i < x.numel; ++i)
            gx[i] = g[i] * (elem_at(x, i, cs) > 0.f ? 1.f : 0.f);
    }

    return grad_x;
}


Tensor relu(const Tensor& x) {
    Tensor out = ReLUOp::forward(x);

    if (grad_mode_enabled && x.requires_grad()) {
        out.autograd_meta = make_grad_meta(
            "ReLUOp",
            {x.autograd_meta},
            [x_save = x](const Tensor& grad) -> std::vector<Tensor> {
                return {ReLUOp::backward(grad, x_save)};
            }
        );
    }

    return out;
}
