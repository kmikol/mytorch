#pragma once

#include <vector>

#include "autograd.h"


/**
 * 2D matrix multiplication: C = A @ B.
 *
 * The forward and backward passes are separated so each can be tested and
 * reasoned about independently. Use the free function matmul() for the
 * differentiable version that wires both into the autograd graph.
 *
 * Both passes are stride-aware: non-contiguous inputs (e.g. tensors produced
 * by T()) are handled correctly without any data copies.
 */
struct MatMulOp {

    /**
     * Compute C = A @ B.
     *
     * WHY strides instead of assuming row-major:
     *   T() does NOT copy data — it just swaps the stride values.
     *   A transposed [M×K] tensor has strides (1, M) instead of (K, 1).
     *   Using A.strides[0]/strides[1] makes the same kernel handle both
     *   forward (contiguous) and backward (transposed) without copies.
     *
     * WHY tiling (the triple m0/n0/k0 loop):
     *   The naive i-j-k loop reads a full column of B for each (m,k) pair.
     *   With K=784 that is 3 KB of stride, evicting the cache on every step.
     *   Tiling keeps a T×T block of A and B resident in L1 at the same time,
     *   so each element loaded from RAM is reused T times before eviction.
     *   T=64 → three tiles of 64×64×4 bytes = 48 KB, fits in a 32–64 KB L1.
     *
     * WHY m-k-n inner order:
     *   The innermost loop increments n, stepping through c[m*N+n] and
     *   b[k*sB0 + n*sB1] contiguously (sB1==1 for a contiguous B). This is
     *   a sequential access pattern — the prefetcher handles it well.
     *   m-n-k would stride through B by sB0 on every step — cache-hostile.
     *
     * WHY hoist a_mk:
     *   a[m*sA0 + k*sA1] doesn't depend on n. Without the hoist the compiler
     *   may re-load it every iteration due to aliasing concerns. Hoisting it
     *   guarantees one load per (m,k) pair instead of one per (m,k,n) triple.
     *
     * Preconditions (asserted at runtime):
     *   - A.ndim == 2, B.ndim == 2
     *   - A.shape[1] == B.shape[0]  (inner dimensions must match)
     *
     * Returns a fresh contiguous [M×N] tensor. Does not touch the autograd graph.
     */
    static Tensor forward(const Tensor& A, const Tensor& B);

    /**
     * Compute input gradients given the upstream gradient G = dL/dC.
     *
     *   dL/dA = G  @ B^T    shape: [M×N] @ [N×K] → [M×K]
     *   dL/dB = A^T @ G     shape: [K×M] @ [M×N] → [K×N]
     *
     * Both gradients are always computed; the autograd engine skips
     * accumulation for inputs that do not require a gradient.
     *
     * @param grad  Upstream gradient (same shape as the forward output, [M×N]).
     * @param A     Left-hand input saved at forward time.
     * @param B     Right-hand input saved at forward time.
     * @return      {grad_A, grad_B}, both contiguous.
     */
    static std::vector<Tensor> backward(const Tensor& grad,
                                        const Tensor& A,
                                        const Tensor& B);
};


/**
 * Differentiable 2D matrix multiply: C = A @ B.
 *
 * Calls MatMulOp::forward for the computation. When at least one input
 * requires a gradient and grad_mode is enabled, registers a backward node
 * so that backward() can propagate gradients through this operation.
 *
 * A and B are captured by value (shared storage) inside the backward closure.
 * No in-place mutations should be made to either tensor after this call.
 */
Tensor matmul(const Tensor& A, const Tensor& B);
