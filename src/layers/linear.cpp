#include "layers/linear.h"

#include <cassert>
#include <cmath>
#include <random>

static Shape make_shape_2d(size_t d0, size_t d1) {
    Shape s{};
    s[0] = d0;
    s[1] = d1;
    return s;
}


Linear::Linear(size_t in_features, size_t out_features)
    : weight(make_shape_2d(in_features, out_features), 2, /*requires_grad=*/true),
      bias(make_shape_2d(1, out_features), 2, /*requires_grad=*/true),
      in_features(in_features),
      out_features(out_features) {
    assert(in_features > 0 && out_features > 0);

    bias.storage->fill(0.f);

    float limit = std::sqrt(6.f / static_cast<float>(in_features + out_features));
    std::uniform_real_distribution<float> dist(-limit, limit);
    static thread_local std::mt19937 rng(std::random_device{}());

    for (size_t i = 0; i < weight.numel; ++i)
        weight.storage->data[i] = dist(rng);
}


Tensor Linear::forward(const Tensor& x) const {
    assert(x.ndim == 2);
    assert(x.shape[1] == in_features);
    return add(matmul(x, weight), bias);
}


std::vector<Tensor*> Linear::parameters() {
    return {&weight, &bias};
}
