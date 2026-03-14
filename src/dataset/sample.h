#pragma once

#include "tensor/tensor.h"

// A single supervised training example.
// By convention each tensor has leading sample dimension = 1:
//   input  shape = [1, input_features]
//   target shape = [1, target_features]
struct Sample {
    Tensor input;
    Tensor target;
};
