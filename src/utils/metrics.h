#pragma once

#include <cstddef>
#include <vector>

struct Metrics {
    float accuracy = 0.0f;
};

Metrics compute_metrics(const std::vector<size_t>& predicted,
                        const std::vector<size_t>& ground_truth);
