#include "utils/metrics.h"

#include <cassert>

Metrics compute_metrics(const std::vector<size_t>& predicted,
                        const std::vector<size_t>& ground_truth) {
    assert(predicted.size() == ground_truth.size());

    Metrics out{};
    if (predicted.empty())
        return out;

    size_t correct = 0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        if (predicted[i] == ground_truth[i])
            ++correct;
    }

    out.accuracy = static_cast<float>(correct) / static_cast<float>(predicted.size());
    return out;
}
