#include "networks/mlp.h"

#include <cassert>
#include <utility>

MLP::MLP(size_t input_features,
         const std::vector<size_t>& hidden_layer_features,
         ActivationFn activation,
         size_t output_features)
    : activation_(std::move(activation)) {
    assert(input_features > 0);
    assert(output_features > 0);
    assert(static_cast<bool>(activation_));

    std::vector<size_t> dims;
    dims.reserve(hidden_layer_features.size() + 2);
    dims.push_back(input_features);

    for (size_t hidden : hidden_layer_features) {
        assert(hidden > 0);
        dims.push_back(hidden);
    }

    dims.push_back(output_features);

    layers_.reserve(dims.size() - 1);
    for (size_t i = 0; i + 1 < dims.size(); ++i)
        layers_.emplace_back(dims[i], dims[i + 1]);
}

Tensor MLP::forward(const Tensor& x) const {
    assert(!layers_.empty());

    Tensor out = x;
    for (size_t i = 0; i < layers_.size(); ++i) {
        out = layers_[i].forward(out);
        if (i + 1 < layers_.size())
            out = activation_(out);
    }

    return out;
}

std::vector<Tensor*> MLP::parameters() {
    std::vector<Tensor*> params;

    for (auto& layer : layers_) {
        std::vector<Tensor*> layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }

    return params;
}
