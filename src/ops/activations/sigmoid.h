#pragma once

#include <vector>

#include "autograd.h"


/**
 * Element-wise sigmoid activation: out = 1 / (1 + exp(-x)).
 *
 * The forward and backward passes are separated so each can be tested and
 * reasoned about independently. Use the free function sigmoid() for the
 * differentiable version that wires both into the autograd graph.
 */
struct SigmoidOp {

    /**
     * Compute out[i] = 1 / (1 + exp(-x[i])) element-wise.
     *
     * Returns a fresh contiguous tensor of the same shape as x.
     * Does not touch the autograd graph.
     */
    static Tensor forward(const Tensor& x);

    /**
     * Compute the input gradient given the upstream gradient.
     *
     *   dL/dx[i] = grad[i] * out[i] * (1 - out[i])
     *
     * WHY the backward takes the forward OUTPUT (not the input):
     *   The sigmoid derivative σ'(x) = σ(x)(1−σ(x)) is most cheaply expressed
     *   in terms of the already-computed output. Re-computing exp(-x) from x
     *   would cost an extra pass; using the saved output costs nothing extra.
     *
     * @param grad  Upstream gradient (same shape as the forward output).
     * @param out   Forward output saved at forward time (i.e. σ(x)).
     * @return      {grad_x} — contiguous, same shape as out.
     */
    static Tensor backward(const Tensor& grad, const Tensor& out);
};


/**
 * Differentiable sigmoid: out = 1 / (1 + exp(-x)).
 *
 * Calls SigmoidOp::forward for the computation. When x requires a gradient
 * and grad_mode is enabled, registers a backward node so that backward() can
 * propagate gradients through this operation.
 *
 * The forward OUTPUT (not the input) is captured inside the backward closure,
 * since the sigmoid derivative is σ(x)(1−σ(x)).
 */
Tensor sigmoid(const Tensor& x);
