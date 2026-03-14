#include "ops/add.h"

#include <cassert>


// Compute the broadcast output shape. For each dim, the output size is
// max(a.shape[d], b.shape[d]). Asserts that the two sizes are compatible
// (equal, or at least one is 1).
static Shape broadcast_shape(const Tensor& a, const Tensor& b) {
    assert(a.ndim == b.ndim);
    Shape out{};
    for (size_t d = 0; d < a.ndim; ++d) {
        assert(a.shape[d] == b.shape[d] || a.shape[d] == 1 || b.shape[d] == 1);
        out[d] = std::max(a.shape[d], b.shape[d]);
    }
    return out;
}

// Decompose a contiguous flat index into per-dim indices, then look up the
// element in t, clamping any broadcast dim to index 0.
static float broadcast_at(const Tensor& t, size_t flat_idx,
                           const Shape& out_shape, size_t ndim) {
    size_t storage_idx = t.offset;
    size_t rem = flat_idx;
    // Compute contiguous strides of the output shape to decompose flat_idx.
    // We iterate dims from outermost to innermost.
    for (size_t d = 0; d < ndim; ++d) {
        // Contiguous stride for dim d in out_shape.
        size_t out_stride = 1;
        for (size_t j = d + 1; j < ndim; ++j) out_stride *= out_shape[j];

        size_t dim_idx = rem / out_stride;
        rem           %= out_stride;

        // Clamp to 0 for broadcast dims.
        if (t.shape[d] == 1) dim_idx = 0;

        storage_idx += dim_idx * t.strides[d];
    }
    return t.storage->data[storage_idx];
}


Tensor AddOp::forward(const Tensor& a, const Tensor& b) {
    Shape out_shape = broadcast_shape(a, b);
    size_t ndim     = a.ndim;

    Tensor out = Tensor::zeros(out_shape, ndim);
    float* o = out.storage->data;  // out is contiguous, offset = 0
    for (size_t i = 0; i < out.numel; ++i)
        o[i] = broadcast_at(a, i, out_shape, ndim)
             + broadcast_at(b, i, out_shape, ndim);

    return out;
}


std::vector<Tensor> AddOp::backward(const Tensor& grad,
                                    const Tensor& a,
                                    const Tensor& b) {
    size_t ndim = grad.ndim;

    Tensor grad_a = Tensor::zeros(a.shape, ndim);
    Tensor grad_b = Tensor::zeros(b.shape, ndim);

    // Compute the contiguous strides of grad's shape (== output shape) so we
    // can decompose each flat grad index into per-dim indices.
    for (size_t i = 0; i < grad.numel; ++i) {
        // Decompose flat index i into per-dim indices using grad's shape.
        size_t rem = i;
        size_t a_flat = 0, b_flat = 0;

        for (size_t d = 0; d < ndim; ++d) {
            size_t out_stride = 1;
            for (size_t j = d + 1; j < ndim; ++j) out_stride *= grad.shape[j];

            size_t dim_idx = rem / out_stride;
            rem           %= out_stride;

            // Map to input index: broadcast dims always map to index 0.
            size_t a_idx = (a.shape[d] == 1) ? 0 : dim_idx;
            size_t b_idx = (b.shape[d] == 1) ? 0 : dim_idx;

            // Contiguous strides for grad_a and grad_b (both freshly allocated,
            // so contiguous strides equal strides_from_shape).
            size_t a_stride = 1, b_stride = 1;
            for (size_t j = d + 1; j < ndim; ++j) {
                a_stride *= a.shape[j];
                b_stride *= b.shape[j];
            }

            a_flat += a_idx * a_stride;
            b_flat += b_idx * b_stride;
        }

        // grad is always contiguous (produced by autograd's clone/zeros).
        float gi = grad.storage->data[i];
        grad_a.storage->data[a_flat] += gi;
        grad_b.storage->data[b_flat] += gi;
    }

    return {grad_a, grad_b};
}


Tensor add(const Tensor& a, const Tensor& b) {
    Tensor out = AddOp::forward(a, b);

    if (grad_mode_enabled && (a.requires_grad() || b.requires_grad())) {
        // Capture a and b by value: Tensor copies share the same
        // shared_ptr<Storage>, so this is a zero-copy view of each input.
        // Safe as long as no in-place mutations are made to a or b after
        // this call.
        out.autograd_meta = make_grad_meta(
            "AddOp",
            {a.autograd_meta, b.autograd_meta},
            [a_save = a, b_save = b](const Tensor& grad) {
                return AddOp::backward(grad, a_save, b_save);
            }
        );
    }

    return out;
}
