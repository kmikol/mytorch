#pragma once

#include <cstddef>
#include <vector>

#include "tensor/tensor.h"

class Linear {
public:
    Linear(size_t in_features, size_t out_features);

    Tensor forward(const Tensor& x) const;
    std::vector<Tensor*> parameters();

    size_t in_features;
    size_t out_features;
    Tensor weight;
    Tensor bias;
};
