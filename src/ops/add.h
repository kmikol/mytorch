#pragma once

#include <vector>

#include "autograd.h"


/**
 * Element-wise tensor addition with optional broadcasting.
 *
 * Broadcasting rules (numpy-style, restricted to matching ndim):
 *   - a.ndim == b.ndim is required.
 *   - For each dimension d, a.shape[d] == b.shape[d]  OR  one of them is 1.
 *   - The output shape is max(a.shape[d], b.shape[d]) for each d.
 *
 * The forward and backward passes are separated so each can be tested and
 * reasoned about independently. Use the free function add() for the
 * differentiable version that wires both into the autograd graph.
 */
struct AddOp {

    /**
     * Compute the element-wise sum with broadcasting: out[...] = a[...] + b[...].
     *
     * WHY flat iteration over the output + per-dim index decomposition:
     *   The output is contiguous, so we iterate its flat indices 0..numel-1.
     *   We decompose each flat index into per-dim indices using the output's
     *   contiguous strides, then clamp each dim-index to 0 for any input whose
     *   shape[d] == 1 (i.e. the broadcast dimension). This is equivalent to
     *   repeating that input along that dimension without copying data.
     *
     * Preconditions (asserted at runtime):
     *   - a.ndim == b.ndim
     *   - For all d: a.shape[d] == b.shape[d]  OR  a.shape[d] == 1  OR  b.shape[d] == 1
     *
     * Returns a fresh contiguous tensor of the broadcast output shape.
     * Does not touch the autograd graph.
     */
    static Tensor forward(const Tensor& a, const Tensor& b);

    /**
     * Compute input gradients given the upstream gradient.
     *
     *   dL/da = grad, summed over dimensions where a was broadcast (a.shape[d] == 1)
     *   dL/db = grad, summed over dimensions where b was broadcast (b.shape[d] == 1)
     *
     * WHY summing over broadcast dims:
     *   Broadcasting repeats the input along a dimension, so multiple output
     *   positions share the same input element. Their gradients must be
     *   accumulated (summed) back onto that single input position.
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
 * Differentiable element-wise add with broadcasting: out = a + b.
 *
 * Calls AddOp::forward for the computation. When at least one input requires
 * a gradient and grad_mode is enabled, registers a backward node so that
 * backward() can propagate gradients through this operation.
 *
 * a and b are captured by value (shared storage) inside the backward closure —
 * no in-place mutations should be made to either tensor after this call.
 */
Tensor add(const Tensor& a, const Tensor& b);
