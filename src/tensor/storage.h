#pragma once
#include <vector>

struct Storage {
    
    std::vector<float> data;

    // Constructors
    Storage() = default;
    Storage(size_t n, float fill = 0.f) : data(n, fill) {}

    // Access to pointer to data
    float* ptr() { return data.data();}
    const float* ptr() const { return data.data(); }
};