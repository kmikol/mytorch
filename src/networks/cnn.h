#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "layers/conv2d.h"
#include "layers/linear.h"

class CNN {
public:
    using ActivationFn = std::function<Tensor(const Tensor&)>;

    CNN(size_t input_channels,
        size_t input_height,
        size_t input_width,
        size_t conv_out_channels,
        size_t kernel_h,
        size_t kernel_w,
        ActivationFn activation,
        size_t output_features,
        size_t stride_h = 1,
        size_t stride_w = 1,
        size_t padding_h = 0,
        size_t padding_w = 0);

    Tensor forward(const Tensor& x) const;
    std::vector<Tensor*> parameters();

private:
    Conv2d conv1_;
    Linear classifier_;
    ActivationFn activation_;
    size_t flattened_features_;
};
