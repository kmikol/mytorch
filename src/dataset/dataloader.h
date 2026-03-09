#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <utility>

#include "dataset.h"

class DataLoader {
public:

    DataLoader(const Dataset& dataset,
               size_t batch_size,
               bool shuffle = true)
        : dataset(dataset),
          batch_size(batch_size),
          shuffle(shuffle),
          position(0),
          rng(1337)
    {
        size_t n = dataset.size();

        indices.resize(n);
        for (size_t i = 0; i < n; ++i)
            indices[i] = i;

        if (shuffle)
            std::shuffle(indices.begin(), indices.end(), rng);
    }

    bool has_next() const {
        return position < indices.size();
    }

    void reset() {
        position = 0;

        if (shuffle)
            std::shuffle(indices.begin(), indices.end(), rng);
    }

    std::pair<Tensor, Tensor> next_batch()
    {
        size_t end = std::min(position + batch_size, indices.size());
        size_t current_batch = end - position;

        std::vector<float> input_buffer;
        std::vector<float> target_buffer;

        int64_t input_rows  = 0;
        int64_t target_rows = 0;

        for (size_t b = 0; b < current_batch; ++b)
        {
            Sample s = dataset.get(indices[position + b]);

            if (b == 0) {
                input_rows  = s.input.shape(0);
                target_rows = s.target.shape(0);

                input_buffer.resize(input_rows * current_batch);
                target_buffer.resize(target_rows * current_batch);
            }

            for (int64_t r = 0; r < input_rows; ++r)
                input_buffer[r + b * input_rows] = s.input.at(r,0);

            for (int64_t r = 0; r < target_rows; ++r)
                target_buffer[r + b * target_rows] = s.target.at(r,0);
        }

        position = end;

        Tensor inputs  = Tensor::from_data(input_buffer,  {input_rows,  (int64_t)current_batch});
        Tensor targets = Tensor::from_data(target_buffer, {target_rows, (int64_t)current_batch});

        return {inputs, targets};
    }

private:

    const Dataset& dataset;

    size_t batch_size;
    size_t position;

    bool shuffle;

    std::vector<size_t> indices;

    std::mt19937 rng;
};