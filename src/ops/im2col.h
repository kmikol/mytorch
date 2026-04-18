#pragma once

#include <cstddef>

#include "tensor/tensor.h"

/**
 * Convert a 4D NCHW input tensor into a 2D patch matrix.
 *
 * Input shape:
 *   x: [N, C, H, W]
 *
 * Output shape:
 *   cols: [N * out_h * out_w, C * kernel_h * kernel_w]
 */
struct Im2ColOp {
    static Tensor forward(const Tensor& x,
                          size_t kernel_h,
                          size_t kernel_w,
                          size_t stride_h = 1,
                          size_t stride_w = 1,
                          size_t pad_h = 0,
                          size_t pad_w = 0);
};

Tensor im2col(const Tensor& x,
              size_t kernel_h,
              size_t kernel_w,
              size_t stride_h = 1,
              size_t stride_w = 1,
              size_t pad_h = 0,
              size_t pad_w = 0);
