#pragma once
#include <cmath>
#include <random>
#include "ops/ops.h"
#include "utils/random.h"

struct Linear {

    Tensor W;   // weights  [out_features, in_features]
    Tensor b;   // bias     [out_features, 1]

    // constructor: set up W and b with the right shapes and initialise
    Linear(int64_t in_features, int64_t out_features) {

        // Xavier uniform initialisation
        // keeps activation variance stable regardless of layer size
        // formula: sample uniformly from [-limit, limit]
        // where limit = sqrt(6 / (in + out))
        float limit = std::sqrt(6.f / (float)(in_features + out_features));
        std::uniform_real_distribution<float> dist(-limit, limit);

        // initialise W with random values, requires_grad so SGD can update it
        W = Tensor::zeros({out_features, in_features}, /*requires_grad=*/true);
        for (int64_t r = 0; r < out_features; r++)
            for (int64_t c = 0; c < in_features; c++)
                W.at(r, c) = dist(global_rng);

        // bias starts at zero — common practice
        // also requires_grad so it gets updated too
        b = Tensor::zeros({out_features, 1}, /*requires_grad=*/true);
    }

    // forward pass: compute W @ x + b
    Tensor forward(const Tensor& x) const {
        return add(matmul(W, x), b);
    }

    // return pointers to all learnable parameters
    // SGD needs these to update them during training
    std::vector<Tensor*> parameters() {
        return {&W, &b};
    }
};