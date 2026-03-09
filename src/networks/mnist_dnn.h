#pragma once
#include "layers/linear.h"
#include "activations/activations.h"
#include "ops/ops.h"

// three-layer DNN for MNIST
// 784 → 256 → 128 → 10
// hidden activations: relu
// output: raw logits (cross_entropy_loss applies softmax internally)
struct MnistDNN {

    Linear l1{784, 128};
    Linear l2{128, 64};
    Linear l3{64,  10};

    Tensor forward(const Tensor& x) {
        // x: [784, N]
        Tensor h1 = relu(l1.forward(x));    // [256, N]
        Tensor h2 = relu(l2.forward(h1));   // [128, N]
        return l3.forward(h2);              // [10,  N] — raw logits
    }

    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> params;
        for (Tensor* p : l1.parameters()) params.push_back(p);
        for (Tensor* p : l2.parameters()) params.push_back(p);
        for (Tensor* p : l3.parameters()) params.push_back(p);
        return params;
    }
};