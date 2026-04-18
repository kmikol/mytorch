#include "ops/activations/sigmoid.h"

#include <cassert>
#include <cmath>


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


Tensor SigmoidOp::forward(const Tensor& x) {
    Tensor out(x.shape, x.ndim);
    float* o = out.storage->data;  // out is contiguous, offset = 0

    if (x.is_contiguous()) {
        const float* xp = x.storage->data + x.offset;
        for (size_t i = 0; i < x.numel; ++i)
            o[i] = 1.f / (1.f + std::exp(-xp[i]));
    } else {
        Strides cs = Tensor::strides_from_shape(x.shape, x.ndim);
        for (size_t i = 0; i < x.numel; ++i)
            o[i] = 1.f / (1.f + std::exp(-elem_at(x, i, cs)));
    }

    return out;
}


Tensor SigmoidOp::backward(const Tensor& grad, const Tensor& out) {
    // grad and out are both contiguous: grad comes from autograd, out was
    // produced by forward() which always allocates fresh contiguous storage.
    assert(grad.is_contiguous());
    assert(out.is_contiguous());

    Tensor grad_x(out.shape, out.ndim);
    const float* g  = grad.storage->data;
    const float* o  = out.storage->data;
    float*       gx = grad_x.storage->data;

    for (size_t i = 0; i < out.numel; ++i)
        gx[i] = g[i] * o[i] * (1.f - o[i]);

    return grad_x;
}


Tensor sigmoid(const Tensor& x) {
    Tensor out = SigmoidOp::forward(x);

    if (grad_mode_enabled && x.requires_grad()) {
        // Capture the forward OUTPUT (not x): the sigmoid derivative
        // σ'(x) = σ(x)(1−σ(x)) is cheapest to compute from σ(x) directly.
        // At capture time out.autograd_meta is still null (set below), so
        // out_save is a clean data-only view sharing the same storage.
        out.autograd_meta = make_grad_meta(
            "SigmoidOp",
            {x.autograd_meta},
            [out_save = out](const Tensor& grad) -> std::vector<Tensor> {
                return {SigmoidOp::backward(grad, out_save)};
            }
        );
    }

    return out;
}
