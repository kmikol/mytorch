#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "layers/linear.h"
#include "tensor/tensor.h"

class MLP {
public:
    using ActivationFn = std::function<Tensor(const Tensor&)>;

    MLP(size_t input_features,
        const std::vector<size_t>& hidden_layer_features,
        ActivationFn activation,
        size_t output_features);

    Tensor forward(const Tensor& x) const;
    std::vector<Tensor*> parameters();

private:
    std::vector<Linear> layers_;
    ActivationFn activation_;
};
