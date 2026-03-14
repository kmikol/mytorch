#include "layers/linear.h"

#include <cassert>
#include <cmath>
#include <random>

#include "ops/add.h"
#include "ops/matmul.h"

namespace {

static Shape make_shape_2d(size_t d0, size_t d1) {
    Shape s{};
    s[0] = d0;
    s[1] = d1;
    return s;
}

}  // namespace

Linear::Linear(size_t in_features, size_t out_features)
    : in_features(in_features),
      out_features(out_features),
      weight(make_shape_2d(in_features, out_features), 2, true),
      bias(Tensor::zeros(make_shape_2d(1, out_features), 2, true)) {
    assert(in_features > 0);
    assert(out_features > 0);

    const float limit = std::sqrt(6.0f / static_cast<float>(in_features + out_features));
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-limit, limit);

    // Fill each weight entry through Tensor indexing (no raw Storage access).
    for (size_t i = 0; i < in_features; ++i)
        for (size_t j = 0; j < out_features; ++j)
            weight(i, j) = dist(rng);
}

Tensor Linear::forward(const Tensor& x) const {
    assert(x.ndim == 2);
    assert(x.shape[1] == in_features);

    return add(matmul(x, weight), bias);
}

std::vector<Tensor*> Linear::parameters() {
    return {&weight, &bias};
}
