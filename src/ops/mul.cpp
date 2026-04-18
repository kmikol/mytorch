#include "ops/mul.h"

#include <cassert>


// Decompose logical flat index i into a storage offset for tensor t,
// using precomputed contiguous strides cs of the shared shape.
static size_t storage_offset(const Tensor& t, size_t i, const Strides& cs) {
    size_t idx = t.offset, rem = i;
    for (size_t d = 0; d < t.ndim; ++d) {
        size_t di  = rem / cs[d];
        rem       %= cs[d];
        idx       += di * t.strides[d];
    }
    return idx;
}


Tensor MulOp::forward(const Tensor& a, const Tensor& b) {
    assert(a.ndim == b.ndim);
    for (size_t d = 0; d < a.ndim; ++d)
        assert(a.shape[d] == b.shape[d]);

    Tensor out(a.shape, a.ndim);
    float*       o  = out.storage->data;          // out is contiguous, offset = 0
    const float* ap = a.storage->data + a.offset;
    const float* bp = b.storage->data + b.offset;

    if (a.is_contiguous() && b.is_contiguous()) {
        for (size_t i = 0; i < a.numel; ++i)
            o[i] = ap[i] * bp[i];
    } else {
        Strides cs = Tensor::strides_from_shape(a.shape, a.ndim);
        for (size_t i = 0; i < a.numel; ++i)
            o[i] = a.storage->data[storage_offset(a, i, cs)]
                 * b.storage->data[storage_offset(b, i, cs)];
    }

    return out;
}


std::vector<Tensor> MulOp::backward(const Tensor& grad,
                                    const Tensor& a,
                                    const Tensor& b) {
    Tensor grad_a(a.shape, a.ndim);
    Tensor grad_b(b.shape, b.ndim);
    float* ga = grad_a.storage->data;
    float* gb = grad_b.storage->data;

    // grad is always contiguous (produced by autograd's clone/zeros).
    assert(grad.is_contiguous());
    const float* g = grad.storage->data;

    if (a.is_contiguous() && b.is_contiguous()) {
        const float* ap = a.storage->data + a.offset;
        const float* bp = b.storage->data + b.offset;
        for (size_t i = 0; i < grad.numel; ++i) {
            ga[i] = g[i] * bp[i];
            gb[i] = g[i] * ap[i];
        }
    } else {
        Strides cs = Tensor::strides_from_shape(a.shape, a.ndim);
        for (size_t i = 0; i < grad.numel; ++i) {
            float gi = g[i];
            ga[i] = gi * b.storage->data[storage_offset(b, i, cs)];
            gb[i] = gi * a.storage->data[storage_offset(a, i, cs)];
        }
    }

    return {grad_a, grad_b};
}


Tensor mul(const Tensor& a, const Tensor& b) {
    Tensor out = MulOp::forward(a, b);

    if (grad_mode_enabled && (a.requires_grad() || b.requires_grad())) {
        // Capture a and b by value: Tensor copies share the same
        // shared_ptr<Storage>, so this is a zero-copy view of each input.
        // Safe as long as no in-place mutations are made to a or b after
        // this call.
        out.autograd_meta = make_grad_meta(
            "MulOp",
            {a.autograd_meta, b.autograd_meta},
            [a_save = a, b_save = b](const Tensor& grad) {
                return MulOp::backward(grad, a_save, b_save);
            }
        );
    }

    return out;
}
