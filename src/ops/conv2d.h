#pragma once

#include <cstddef>
#include <vector>

#include "autograd.h"

struct Conv2dOp {
    static Tensor forward(const Tensor& x,
                          const Tensor& weight,
                          const Tensor& bias,
                          size_t stride_h = 1,
                          size_t stride_w = 1,
                          size_t pad_h = 0,
                          size_t pad_w = 0);

    static std::vector<Tensor> backward(const Tensor& grad,
                                        const Tensor& x,
                                        const Tensor& weight,
                                        const Tensor& bias,
                                        size_t stride_h = 1,
                                        size_t stride_w = 1,
                                        size_t pad_h = 0,
                                        size_t pad_w = 0);
};

Tensor conv2d(const Tensor& x,
              const Tensor& weight,
              const Tensor& bias,
              size_t stride_h = 1,
              size_t stride_w = 1,
              size_t pad_h = 0,
              size_t pad_w = 0);
