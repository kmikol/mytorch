#pragma once
#include <cstddef>
#include "sample.h"

class Dataset {
public:
    virtual ~Dataset() = default;

    virtual size_t size() const = 0;
    virtual Sample get(size_t index) const = 0;
};