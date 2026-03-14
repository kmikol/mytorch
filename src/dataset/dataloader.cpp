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
      rng(seed),
      input_dim_(dataset.input_dim()),
      target_dim_(dataset.target_dim()) {
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

    const size_t end = std::min(position + batch_size, indices.size());
    const size_t B   = end - position;

    // Pre-allocate batch tensors once — no per-sample Tensor allocation.
    Tensor inputs  = Tensor::zeros(make_shape_2d(B, input_dim_),  2);
    Tensor targets = Tensor::zeros(make_shape_2d(B, target_dim_), 2);

    float* inp = inputs.storage->data;
    float* tgt = targets.storage->data;

    for (size_t b = 0; b < B; ++b)
        dataset.fill_sample(indices[position + b],
                            inp + b * input_dim_,
                            tgt + b * target_dim_);

    position = end;
    return {inputs, targets};
}
