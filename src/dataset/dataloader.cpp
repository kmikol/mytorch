#include "dataset/dataloader.h"

#include <algorithm>
#include <cassert>


static Shape make_shape_2d(size_t d0, size_t d1) {
    Shape s{};
    s[0] = d0;
    s[1] = d1;
    return s;
}


DataLoader::DataLoader(const Dataset& dataset,
                       size_t batch_size,
                       bool shuffle,
                       uint32_t seed)
    : dataset(dataset),
      batch_size(batch_size),
      shuffle(shuffle),
      position(0),
      rng(seed) {
    assert(batch_size > 0);

    size_t n = dataset.size();
    indices.resize(n);
    for (size_t i = 0; i < n; ++i)
        indices[i] = i;

    if (shuffle)
        std::shuffle(indices.begin(), indices.end(), rng);
}


bool DataLoader::has_next() const {
    return position < indices.size();
}


void DataLoader::reset() {
    position = 0;
    if (shuffle)
        std::shuffle(indices.begin(), indices.end(), rng);
}


std::pair<Tensor, Tensor> DataLoader::next_batch() {
    assert(has_next());

    size_t end = std::min(position + batch_size, indices.size());
    size_t current_batch = end - position;

    Sample first = dataset.get(indices[position]);
    assert(first.input.ndim == 2 && first.target.ndim == 2);
    assert(first.input.shape[0] == 1 && first.target.shape[0] == 1);

    size_t input_features = first.input.shape[1];
    size_t target_features = first.target.shape[1];

    Tensor inputs = Tensor::zeros(make_shape_2d(current_batch, input_features), 2);
    Tensor targets = Tensor::zeros(make_shape_2d(current_batch, target_features), 2);

    for (size_t b = 0; b < current_batch; ++b) {
        Sample s = dataset.get(indices[position + b]);

        assert(s.input.ndim == 2 && s.target.ndim == 2);
        assert(s.input.shape[0] == 1 && s.target.shape[0] == 1);
        assert(s.input.shape[1] == input_features);
        assert(s.target.shape[1] == target_features);

        for (size_t f = 0; f < input_features; ++f)
            inputs(b, f) = s.input(0, f);

        for (size_t f = 0; f < target_features; ++f)
            targets(b, f) = s.target(0, f);
    }

    position = end;
    return {inputs, targets};
}
