#include "loss_functions/cross_entropy.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>


Tensor CrossEntropyOp::forward(const Tensor& logits, const Tensor& targets) {
    assert(logits.ndim == 2 && targets.ndim == 2);
    assert(logits.shape[0] == targets.shape[0] && logits.shape[1] == targets.shape[1]);

    const size_t N = logits.shape[0];
    const size_t C = logits.shape[1];

    float total_loss = 0.f;

    for (size_t i = 0; i < N; ++i) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < C; ++j)
            row_max = std::max(row_max, logits(i, j));

        float sum_exp = 0.f;
        for (size_t j = 0; j < C; ++j)
            sum_exp += std::exp(logits(i, j) - row_max);

        const float logsumexp = row_max + std::log(sum_exp);
        for (size_t j = 0; j < C; ++j)
            total_loss += -targets(i, j) * (logits(i, j) - logsumexp);
    }

    const float loss = total_loss / static_cast<float>(N);

    Shape out_shape{};
    out_shape[0] = 1;
    Tensor out = Tensor::zeros(out_shape, 1);
    out.storage->data[0] = loss;
    return out;
}


std::vector<Tensor> CrossEntropyOp::backward(const Tensor& grad,
                                              const Tensor& logits,
                                              const Tensor& targets) {
    assert(grad.is_contiguous());
    const float upstream = grad.storage->data[0];  // scalar upstream gradient

    const size_t N = logits.shape[0];
    const size_t C = logits.shape[1];

    Tensor grad_logits = Tensor::zeros(logits.shape, logits.ndim);

    for (size_t i = 0; i < N; ++i) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < C; ++j)
            row_max = std::max(row_max, logits(i, j));

        float sum_exp = 0.f;
        for (size_t j = 0; j < C; ++j)
            sum_exp += std::exp(logits(i, j) - row_max);

        for (size_t j = 0; j < C; ++j) {
            const float softmax = std::exp(logits(i, j) - row_max) / sum_exp;
            grad_logits(i, j) = (upstream / static_cast<float>(N)) * (softmax - targets(i, j));
        }
    }

    return {grad_logits};
}


Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
    Tensor out = CrossEntropyOp::forward(logits, targets);

    if (grad_mode_enabled && logits.requires_grad()) {
        // Only logits participates in the graph — targets are constant data.
        out.autograd_meta = make_grad_meta(
            "CrossEntropyOp",
            {logits.autograd_meta},
            [l_save = logits, t_save = targets](const Tensor& grad) {
                return CrossEntropyOp::backward(grad, l_save, t_save);
            }
        );
    }

    return out;
}
