#include "ops/reshape.h"

#include <cassert>

Tensor ReshapeOp::forward(const Tensor& x, const Shape& new_shape, size_t new_ndim) {
    assert(new_ndim > 0 && new_ndim <= MAX_DIM);
    assert(x.is_contiguous());
    assert(x.offset == 0);

    size_t new_numel = 1;
    for (size_t i = 0; i < new_ndim; ++i)
        new_numel *= new_shape[i];
    assert(new_numel == x.numel);

    Tensor out = x;
    out.autograd_meta = nullptr;
    out.shape = new_shape;
    out.ndim = new_ndim;
    out.numel = new_numel;
    out.strides = Tensor::strides_from_shape(new_shape, new_ndim);
    return out;
}

std::vector<Tensor> ReshapeOp::backward(const Tensor& grad,
                                        const Shape& old_shape,
                                        size_t old_ndim) {
    return {ReshapeOp::forward(grad, old_shape, old_ndim)};
}

Tensor reshape(const Tensor& x, const Shape& new_shape, size_t new_ndim) {
    Tensor out = ReshapeOp::forward(x, new_shape, new_ndim);

    if (grad_mode_enabled && x.requires_grad()) {
        out.autograd_meta = make_grad_meta(
            "reshape",
            {x.autograd_meta},
            [old_shape = x.shape, old_ndim = x.ndim](const Tensor& grad) {
                return ReshapeOp::backward(grad, old_shape, old_ndim);
            }
        );
    }

    return out;
}
