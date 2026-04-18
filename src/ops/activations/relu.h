#pragma once

#include <vector>

#include "autograd.h"


/**
 * Element-wise Rectified Linear Unit activation: out = max(0, x).
 *
 * The forward and backward passes are separated so each can be tested and
 * reasoned about independently. Use the free function relu() for the
 * differentiable version that wires both into the autograd graph.
 */
struct ReLUOp {

    /**
     * Compute out[i] = max(0, x[i]) element-wise.
     *
     * Returns a fresh contiguous tensor of the same shape as x.
     * Does not touch the autograd graph.
     */
    static Tensor forward(const Tensor& x);

    /**
     * Compute the input gradient given the upstream gradient.
     *
     *   dL/dx[i] = grad[i]  if x[i] > 0
     *              0         otherwise
     *
     * The mask is recomputed from the saved input x rather than stored
     * separately, keeping memory use minimal.
     *
     * @param grad  Upstream gradient (same shape as the forward output).
     * @param x     Input saved at forward time.
     * @return      {grad_x} — contiguous, same shape as x.
     */
    static Tensor backward(const Tensor& grad, const Tensor& x);
};


/**
 * Differentiable ReLU: out = max(0, x).
 *
 * Calls ReLUOp::forward for the computation. When x requires a gradient and
 * grad_mode is enabled, registers a backward node so that backward() can
 * propagate gradients through this operation.
 *
 * x is captured by value (shared storage) inside the backward closure —
 * no in-place mutations should be made to x after this call.
 */
Tensor relu(const Tensor& x);
