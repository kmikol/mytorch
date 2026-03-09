#pragma once
#include "tensorlib.h"
#include "autograd.h"
#include <cmath>
#include <limits>

// targets tensor format: [1, N] storing float-cast integer class indices
// e.g. three samples with classes 0, 2, 1 → from_data({0, 2, 1}, {1, 3})
// we don't have an integer tensor type so class indices are stored as floats

struct CrossEntropyLossOp {

    // ----------------------------------------------------------------
    // forward
    //
    // fuses softmax + negative log likelihood into one pass
    // this avoids computing log(softmax(x)) directly, which would
    // require storing the softmax output and risks log(~0) = -inf
    //
    // instead we use the log-sum-exp formulation:
    //   loss_n = -logits[target_n, n] + log(sum_c(exp(logits[c,n])))
    //
    // with max subtraction for numerical stability:
    //   loss_n = -logits[target_n, n] + max_n
    //            + log(sum_c(exp(logits[c,n] - max_n)))
    //
    // logits:  [C, N]  — raw unnormalised scores, one per class per sample
    // targets: [1, N]  — class index for each sample, stored as float
    // returns: [1, 1]  — scalar mean loss
    // ----------------------------------------------------------------
    static Tensor forward(const Tensor& logits, const Tensor& targets) {
        int64_t C = logits.shape(0);   // number of classes
        int64_t N = logits.shape(1);   // batch size

        float total_loss = 0.f;

        for (int64_t n = 0; n < N; n++) {

            // step 1: find max logit for this sample — stability trick
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t c = 0; c < C; c++) {
                float v = logits.at({c, n});
                if (v > max_val) max_val = v;
            }

            // step 2: compute log(sum(exp(logits - max)))
            // subtracting max makes the largest exp term = exp(0) = 1
            // all other terms <= 1, so no overflow
            float sum_exp = 0.f;
            for (int64_t c = 0; c < C; c++)
                sum_exp += std::exp(logits.at({c, n}) - max_val);

            // log(sum_exp) + max_val = log(sum(exp(logits)))
            // the max_val we subtracted earlier is added back here
            float log_normaliser = std::log(sum_exp) + max_val;

            // step 3: NLL for this sample
            // loss = -logits[target] + log(sum(exp(logits)))
            int64_t t = (int64_t)targets.at({0, n});
            assert(t >= 0 && t < C && "target class index out of range");

            total_loss += -logits.at({t, n}) + log_normaliser;
        }

        // mean over batch
        return Tensor::from_data({total_loss / (float)N}, {1, 1});
    }

    // ----------------------------------------------------------------
    // backward
    //
    // the gradient of fused softmax + cross-entropy is:
    //   d_logits[i, n] = (softmax(logits)[i, n] - one_hot[target_n][i]) / N
    //
    // derivation sketch:
    //   loss = -logits[t,n] + log(sum_c exp(logits[c,n]))
    //   d_loss/d_logits[i,n] = -1[i==t] + exp(logits[i,n]) / sum_c exp(...)
    //                        = -one_hot[i] + softmax[i]
    //                        = softmax[i] - one_hot[i]
    //   divide by N for the mean
    //
    // we recompute softmax from saved logits rather than saving the
    // probabilities separately — trades a small recompute for one fewer
    // [C,N] tensor kept alive between forward and backward
    // ----------------------------------------------------------------
    static std::vector<Tensor> backward(
        const Tensor& grad,            // [1,1] upstream scalar
        const Tensor& saved_logits,    // [C, N]
        const Tensor& saved_targets)   // [1, N]
    {
        int64_t C = saved_logits.shape(0);
        int64_t N = saved_logits.shape(1);

        float upstream = grad.at({0, 0});

        // recompute softmax probabilities from saved logits
        // same numerically stable computation as forward
        Tensor probs = Tensor::zeros({C, N});

        for (int64_t n = 0; n < N; n++) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t c = 0; c < C; c++) {
                float v = saved_logits.at({c, n});
                if (v > max_val) max_val = v;
            }
            float sum_exp = 0.f;
            for (int64_t c = 0; c < C; c++) {
                float e = std::exp(saved_logits.at({c, n}) - max_val);
                probs.at({c, n}) = e;
                sum_exp += e;
            }
            for (int64_t c = 0; c < C; c++)
                probs.at({c, n}) /= sum_exp;
        }

        // gradient = (softmax - one_hot) / N * upstream
        // start from probs and subtract 1 at each target position
        Tensor d_logits = probs.clone();

        for (int64_t n = 0; n < N; n++) {
            int64_t t = (int64_t)saved_targets.at({0, n});
            // one_hot contribution: subtract 1 at the correct class
            d_logits.at({t, n}) -= 1.f;
        }

        // scale by 1/N (from the mean) and the upstream gradient
        float scale = upstream / (float)N;
        for (int64_t c = 0; c < C; c++)
            for (int64_t n = 0; n < N; n++)
                d_logits.at({c, n}) *= scale;

        // targets never require grad — class indices are not differentiable
        return {d_logits};
    }
};

// ----------------------------------------------------------------
// free function — follows mse_loss conventions:
//   - saves with .clone()
//   - checks requires_grad directly
//   - lambda defined inline
//   - make_grad_meta called at the end
// ----------------------------------------------------------------
inline Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
    assert(logits.ndim()  == 2 && "logits must be [C, N]");
    assert(targets.ndim() == 2 && "targets must be [1, N]");
    assert(targets.shape(0) == 1 && "targets must have shape [1, N]");
    assert(logits.shape(1) == targets.shape(1) && "batch size must match");

    Tensor loss = CrossEntropyLossOp::forward(logits, targets);

    if (grad_mode_enabled && logits.requires_grad()) {
        std::shared_ptr<AutogradMeta> autograd_meta_logits = logits.autograd_meta;

        Tensor saved_logits  = logits.clone();
        Tensor saved_targets = targets.clone();

        std::function<std::vector<Tensor>(const Tensor&)> backward_fn =
            [saved_logits, saved_targets]
            (const Tensor& grad) -> std::vector<Tensor> {
                return CrossEntropyLossOp::backward(
                    grad, saved_logits, saved_targets);
            };

        loss.autograd_meta = make_grad_meta(
            "cross_entropy_loss",
            {autograd_meta_logits},
            backward_fn);
    }

    return loss;
}