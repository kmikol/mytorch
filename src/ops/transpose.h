#pragma once
#include <vector>
#include <memory>
#include "tensorlib.h"


struct TransposeOp {

    // forward: zero-copy view with swapped strides
    // dim0 and dim1 are the dimensions being swapped
    static Tensor forward(const Tensor& t, int dim0, int dim1) {
        auto new_shape   = t.implementation->get_shape();
        auto new_strides = t.implementation->get_strides();

        std::swap(new_shape[dim0],   new_shape[dim1]);
        std::swap(new_strides[dim0], new_strides[dim1]);

        Tensor result;
        result.implementation = std::make_shared<TensorImpl>(
            t.implementation->storage,   // shared storage — zero copy
            new_shape,
            new_strides,
            t.implementation->offset
        );
        return result;
    }

    // backward: transpose is its own inverse
    // swapping the same two dims again restores the original layout
    // gradient values pass through unchanged — only layout flips back
    static std::vector<Tensor> backward(
        const Tensor& grad,
        int dim0, int dim1)
    {
        return { TransposeOp::forward(grad, dim0, dim1) };
    }
};

inline Tensor transpose(const Tensor& t, int dim0, int dim1) {
    if (dim0 < 0) dim0 += t.ndim();
    if (dim1 < 0) dim1 += t.ndim();
    assert(dim0 < t.ndim() && dim1 < t.ndim());

    Tensor out = TransposeOp::forward(t, dim0, dim1);

    if (t.requires_grad()) {
        out.autograd_meta = make_grad_meta(
            "transpose",
            {t.autograd_meta},
            [dim0, dim1](const Tensor& grad) {
                return TransposeOp::backward(grad, dim0, dim1);
            });
    }

    return out;
}