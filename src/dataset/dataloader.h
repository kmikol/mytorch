#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "dataset/dataset.h"


class DataLoader {
public:
    DataLoader(const Dataset& dataset,
               size_t batch_size,
               bool shuffle = true,
               uint32_t seed = 1337u);

    bool has_next() const;
    void reset();

    // Returns a pair {inputs, targets} with shapes:
    //   inputs  = [B, input_features]
    //   targets = [B, target_features]
    // where B <= batch_size for the final partial batch.
    std::pair<Tensor, Tensor> next_batch();

private:
    const Dataset& dataset;
    size_t batch_size;
    bool shuffle;

    std::vector<size_t> indices;
    size_t position;
    std::mt19937 rng;

    // Cached at construction via dataset.input_dim() / target_dim() so that
    // next_batch() can pre-allocate without calling dataset.get() at all.
    size_t input_dim_;
    size_t target_dim_;
};
