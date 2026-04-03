#pragma once

#include <cstddef>
#include <vector>

#include "tensor/tensor.h"

class Conv2d {
public:
    Conv2d(size_t in_channels,
           size_t out_channels,
           size_t kernel_h,
           size_t kernel_w,
           size_t stride_h = 1,
           size_t stride_w = 1,
           size_t padding_h = 0,
           size_t padding_w = 0);

    Tensor forward(const Tensor& x) const;
    std::vector<Tensor*> parameters();

    size_t in_channels;
    size_t out_channels;
    size_t kernel_h;
    size_t kernel_w;
    size_t stride_h;
    size_t stride_w;
    size_t padding_h;
    size_t padding_w;

    Tensor weight;  // [out_channels, in_channels, kernel_h, kernel_w]
    Tensor bias;    // [1, out_channels]
};
