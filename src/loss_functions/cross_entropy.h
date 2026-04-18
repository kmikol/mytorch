#pragma once

#include <vector>

#include "autograd.h"


/**
 * Mean categorical cross-entropy from logits.
 *
 * This matches PyTorch-style cross entropy behavior conceptually: it fuses a
 * numerically stable softmax with negative log-likelihood, instead of requiring
 * pre-softmax probabilities as input.
 *
 * The forward and backward passes are separated so each can be tested and
 * reasoned about independently. Use the free function cross_entropy() for the
 * differentiable version that wires both into the autograd graph.
 *
 * Inputs:
 *   logits  — [N, C] raw, unnormalized class scores.
 *   targets — [N, C] ground-truth labels (typically one-hot, same shape).
 *
 * Output scalar [1]:
 *   L = -1/N * Σ_i Σ_j targets[i,j] * log_softmax(logits)[i,j]
 */
struct CrossEntropyOp {

    /**
    * Compute the mean cross-entropy loss from logits.
     *
     * Preconditions (asserted at runtime):
    *   - logits.ndim == 2, targets.ndim == 2
    *   - logits.shape == targets.shape
     *
     * Returns a contiguous [1] tensor holding the scalar loss value.
     * Does not touch the autograd graph.
     */
    static Tensor forward(const Tensor& logits, const Tensor& targets);

    /**
    * Compute the gradient of the loss with respect to logits.
     *
    *   dL/dlogits = upstream * (softmax(logits) - targets) / N
     *
    * WHY only logits gets a gradient:
     *   targets are ground-truth labels — fixed data, not learnable parameters.
     *   Computing their gradient would be meaningless and is skipped entirely.
     *
     * @param grad     Upstream gradient (scalar [1] tensor).
    * @param logits   Raw scores saved at forward time.
     * @param targets  Ground-truth labels saved at forward time.
    * @return         {grad_logits} — contiguous [N, C] tensor.
     */
    static std::vector<Tensor> backward(const Tensor& grad,
                                const Tensor& logits,
                                        const Tensor& targets);
};


/**
 * Differentiable mean categorical cross-entropy loss.
 *
 * Calls CrossEntropyOp::forward for the computation. When logits requires a
 * gradient and grad_mode is enabled, registers a backward node so that
 * backward() can propagate gradients through this operation.
 *
 * Only logits participates in the autograd graph — targets are treated as
 * constant data and never receive a gradient.
 *
 * logits and targets are captured by value (shared storage) inside the backward
 * closure — no in-place mutations should be made to either after this call.
 */
Tensor cross_entropy(const Tensor& logits, const Tensor& targets);
