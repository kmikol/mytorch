#pragma once

#include <cstddef>
#include <vector>

#include "ops/add.h"
#include "ops/matmul.h"


/**
 * Fully connected (affine) layer for 2D batched inputs.
 *
 * Input convention:
 *   x shape = [N, in_features], where N is the number of samples.
 *
 * Parameters:
 *   weight shape = [in_features, out_features]
 *   bias   shape = [1, out_features] (broadcast across batch dimension)
 *
 * Forward:
 *   y = x @ weight + bias
 *   y shape = [N, out_features]
 */
struct Linear {
    Tensor weight;
    Tensor bias;

    size_t in_features;
    size_t out_features;

    Linear(size_t in_features, size_t out_features);

    Tensor forward(const Tensor& x) const;

    std::vector<Tensor*> parameters();
};
