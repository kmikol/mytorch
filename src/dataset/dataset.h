#pragma once

#include <cstddef>

#include "dataset/sample.h"


class Dataset {
public:
    virtual ~Dataset() = default;

    virtual size_t size()       const = 0;
    virtual Sample get(size_t index) const = 0;

    // Flat feature counts — used by DataLoader to pre-allocate batch tensors.
    virtual size_t input_dim()  const = 0;
    virtual size_t target_dim() const = 0;

    // Writes one sample directly into caller-owned float buffers.
    // Avoids the per-sample Tensor allocation that get() incurs.
    // input_buf  must hold input_dim()  floats.
    // target_buf must hold target_dim() floats.
    virtual void fill_sample(size_t index,
                             float* input_buf,
                             float* target_buf) const = 0;
};
