#pragma once

#include <vector>

#include "autograd.h"


/**
 * Mean categorical cross-entropy loss.
 *
 * Expects the predictions to already be probability distributions (i.e. after
 * softmax). Targets may be one-hot or soft labels — any non-negative float
 * matrix whose rows sum to 1.
 *
 * The forward and backward passes are separated so each can be tested and
 * reasoned about independently. Use the free function cross_entropy() for the
 * differentiable version that wires both into the autograd graph.
 *
 * Inputs:
 *   probs   — [N, C] predicted probabilities (rows must be valid distributions).
 *   targets — [N, C] ground-truth labels (same shape as probs).
 *
 * Output:
 *   Scalar [1] tensor:  L = -1/N * Σ_i Σ_j targets[i,j] * log(probs[i,j] + ε)
 *   where ε = 1e-7 guards against log(0).
 */
struct CrossEntropyOp {

    /**
     * Compute the mean cross-entropy loss.
     *
     * Preconditions (asserted at runtime):
     *   - probs.ndim == 2, targets.ndim == 2
     *   - probs.shape == targets.shape
     *
     * Returns a contiguous [1] tensor holding the scalar loss value.
     * Does not touch the autograd graph.
     */
    static Tensor forward(const Tensor& probs, const Tensor& targets);

    /**
     * Compute the gradient of the loss with respect to probs.
     *
     *   dL/dprobs[i,j] = -upstream * targets[i,j] / (probs[i,j] + ε) / N
     *
     * WHY only probs gets a gradient:
     *   targets are ground-truth labels — fixed data, not learnable parameters.
     *   Computing their gradient would be meaningless and is skipped entirely.
     *
     * @param grad     Upstream gradient (scalar [1] tensor).
     * @param probs    Probability predictions saved at forward time.
     * @param targets  Ground-truth labels saved at forward time.
     * @return         {grad_probs} — contiguous [N, C] tensor.
     */
    static std::vector<Tensor> backward(const Tensor& grad,
                                        const Tensor& probs,
                                        const Tensor& targets);
};


/**
 * Differentiable mean categorical cross-entropy loss.
 *
 * Calls CrossEntropyOp::forward for the computation. When probs requires a
 * gradient and grad_mode is enabled, registers a backward node so that
 * backward() can propagate gradients through this operation.
 *
 * Only probs participates in the autograd graph — targets are treated as
 * constant data and never receive a gradient.
 *
 * probs and targets are captured by value (shared storage) inside the backward
 * closure — no in-place mutations should be made to either after this call.
 */
Tensor cross_entropy(const Tensor& probs, const Tensor& targets);
