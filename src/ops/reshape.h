#pragma once

#include <cstddef>
#include <vector>

#include "autograd.h"

struct ReshapeOp {
    static Tensor forward(const Tensor& x, const Shape& new_shape, size_t new_ndim);
    static std::vector<Tensor> backward(const Tensor& grad,
                                        const Shape& old_shape,
                                        size_t old_ndim);
};

Tensor reshape(const Tensor& x, const Shape& new_shape, size_t new_ndim);
