#pragma once

#include <vector>

#include "autograd.h"


/**
 * Element-wise tensor multiplication.
 *
 * The forward and backward passes are separated so each can be tested and
 * reasoned about independently. Use the free function mul() for the
 * differentiable version that wires both into the autograd graph.
 */
struct MulOp {

    /**
     * Compute the element-wise product: out[i] = a[i] * b[i].
     *
     * Preconditions (asserted at runtime):
     *   - a.ndim == b.ndim
     *   - a.shape[d] == b.shape[d] for every dimension d
     *
     * Returns a fresh contiguous tensor. Does not touch the autograd graph.
     */
    static Tensor forward(const Tensor& a, const Tensor& b);

    /**
     * Compute input gradients given the upstream gradient.
     *
     *   dL/da = grad ⊙ b
     *   dL/db = grad ⊙ a
     *
     * @param grad  Upstream gradient (same shape as the forward output).
     * @param a     Left-hand input saved at forward time.
     * @param b     Right-hand input saved at forward time.
     * @return      {grad_a, grad_b} — contiguous, same shape as their inputs.
     */
    static std::vector<Tensor> backward(const Tensor& grad,
                                        const Tensor& a,
                                        const Tensor& b);
};


/**
 * Differentiable element-wise multiply: out = a * b.
 *
 * Calls MulOp::forward for the computation. When at least one input requires
 * a gradient and grad_mode is enabled, registers a backward node so that
 * backward() can propagate gradients through this operation.
 *
 * a and b are captured by value (shared storage) inside the backward closure —
 * no in-place mutations should be made to either tensor after this call.
 */
Tensor mul(const Tensor& a, const Tensor& b);
