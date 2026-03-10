// networks/mnist_dnn.h
#pragma once
#include "layers/linear.h"
#include "ops/ops.h"
#include "optim.h"

class MnistDNN {
public:
    // input_size: number of features per sample (784 for 28x28, 196 for 14x14)
    // num_classes: number of output classes (10 for MNIST)
    explicit MnistDNN(int64_t input_size = 784, int64_t num_classes = 10)
        : first_layer(input_size, 128)
        , second_layer(128, 64)
        , output_layer(64, num_classes)
    {}

    Tensor forward(const Tensor& input) const {
        Tensor hidden_one    = relu(first_layer.forward(input));
        Tensor hidden_two    = relu(second_layer.forward(hidden_one));
        Tensor logits        = output_layer.forward(hidden_two);
        return logits;
    }

    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> all_parameters;
        for (Tensor* parameter : first_layer.parameters())  all_parameters.push_back(parameter);
        for (Tensor* parameter : second_layer.parameters()) all_parameters.push_back(parameter);
        for (Tensor* parameter : output_layer.parameters()) all_parameters.push_back(parameter);
        return all_parameters;
    }

private:
    Linear first_layer;
    Linear second_layer;
    Linear output_layer;
};