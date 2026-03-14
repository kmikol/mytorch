#include "loss_functions/cross_entropy.h"

#include <cassert>
#include <cmath>


static constexpr float kEps = 1e-7f;


Tensor CrossEntropyOp::forward(const Tensor& probs, const Tensor& targets) {
    assert(probs.ndim == 2 && targets.ndim == 2);
    assert(probs.shape[0] == targets.shape[0] && probs.shape[1] == targets.shape[1]);

    size_t N = probs.shape[0];
    size_t C = probs.shape[1];

    float loss = 0.f;
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < C; ++j)
            loss -= targets(i, j) * std::log(probs(i, j) + kEps);

    loss /= static_cast<float>(N);

    Shape out_shape{};
    out_shape[0] = 1;
    Tensor out = Tensor::zeros(out_shape, 1);
    out.storage->data[0] = loss;
    return out;
}


std::vector<Tensor> CrossEntropyOp::backward(const Tensor& grad,
                                              const Tensor& probs,
                                              const Tensor& targets) {
    assert(grad.is_contiguous());
    float upstream = grad.storage->data[0];  // scalar upstream gradient

    size_t N = probs.shape[0];
    size_t C = probs.shape[1];
    float scale = -upstream / static_cast<float>(N);

    Tensor grad_probs = Tensor::zeros(probs.shape, probs.ndim);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < C; ++j)
            grad_probs(i, j) = scale * targets(i, j) / (probs(i, j) + kEps);

    return {grad_probs};
}


Tensor cross_entropy(const Tensor& probs, const Tensor& targets) {
    Tensor out = CrossEntropyOp::forward(probs, targets);

    if (grad_mode_enabled && probs.requires_grad()) {
        // Only probs participates in the graph — targets are constant data.
        out.autograd_meta = make_grad_meta(
            "CrossEntropyOp",
            {probs.autograd_meta},
            [p_save = probs, t_save = targets](const Tensor& grad) {
                return CrossEntropyOp::backward(grad, p_save, t_save);
            }
        );
    }

    return out;
}
