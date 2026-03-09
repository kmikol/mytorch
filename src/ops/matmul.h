#pragma once
#include <vector>
#include "tensorlib.h"

// forward declaration so backward() can call matmul() recursively
// (backward needs matmul before the full definition appears below)
inline Tensor matmul(const Tensor& A, const Tensor& B);

struct MatMulOp {

    // ----------------------------------------------------------------
    // forward: compute C = A @ B
    //
    // WHY strides instead of assuming row-major:
    //   transpose() does NOT copy data — it just swaps the stride values.
    //   A transposed [M x K] tensor has strides (1, M) instead of (K, 1).
    //   If we hardcoded c[m*K + k] we'd read wrong memory for transposed inputs.
    //   Using A.stride(0)/stride(1) makes the same code handle both forward
    //   (contiguous) and backward (transposed) without any branching or copies.
    //
    // WHY tiling (the triple m0/n0/k0 loop):
    //   The naive i-j-k loop reads a full column of B for each (m,k) pair.
    //   A column of B is K floats apart in memory — for K=784 that's 3KB of
    //   stride, so every column access evicts the previous one from L1 cache.
    //   Tiling keeps a T×T block of A and B resident in L1 at the same time,
    //   so each float loaded from RAM is reused T times before being evicted.
    //   T=64 → three tiles of 64×64×4 bytes = 48KB, fits in a typical 32-64KB L1.
    //
    // WHY i-k-j inner order (not i-j-k):
    //   The innermost loop increments n, which steps through c[m*N + n] and
    //   b[k*sB0 + n*sB1] contiguously (sB1==1 for a contiguous or forward-pass B).
    //   This is a sequential memory access pattern — the CPU prefetcher handles it.
    //   i-j-k would instead increment k in the inner loop, causing a stride-N
    //   jump through B on every step, which is cache-hostile for large N.
    //
    // WHY hoist a_mk out of the n loop:
    //   a[m*sA0 + k*sA1] doesn't depend on n — without the hoist the compiler
    //   may recompute or re-load it every iteration due to aliasing concerns
    //   (it can't prove c[] and a[] don't overlap). Hoisting it explicitly
    //   ensures one load per (m,k) pair instead of one load per (m,k,n) triple.
    // ----------------------------------------------------------------
    static Tensor forward(const Tensor& A, const Tensor& B) {
        int64_t M = A.shape(0), K = A.shape(1), N = B.shape(1);

        // allocate output — zero-initialised so += accumulation is correct
        Tensor C = Tensor::zeros({M, N});
        float*       c  = C.data_ptr();
        const float* a  = A.data_ptr();
        const float* b  = B.data_ptr();

        // cache strides once — avoids repeated virtual/shared_ptr dispatch
        // for a contiguous tensor:   sA0=K, sA1=1  (row-major)
        // for a transposed tensor:   sA0=1, sA1=K  (column-major, no copy)
        int64_t sA0 = A.stride(0), sA1 = A.stride(1);
        int64_t sB0 = B.stride(0), sB1 = B.stride(1);

        constexpr int64_t T = 256;  // tile size — tune to L1 cache size

        for (int64_t m0 = 0; m0 < M; m0 += T)
        for (int64_t n0 = 0; n0 < N; n0 += T)
        for (int64_t k0 = 0; k0 < K; k0 += T) {

            // clamp tile edges — handles dimensions not divisible by T
            int64_t m_end = std::min(m0 + T, M);
            int64_t n_end = std::min(n0 + T, N);
            int64_t k_end = std::min(k0 + T, K);

            for (int64_t m = m0; m < m_end; ++m)
            for (int64_t k = k0; k < k_end; ++k) {
                float a_mk = a[m*sA0 + k*sA1];  // hoisted — one load per (m,k)
                for (int64_t n = n0; n < n_end; ++n)
                    c[m*N + n] += a_mk * b[k*sB0 + n*sB1];  // sequential in n
            }
        }
        return C;
    }

    // ----------------------------------------------------------------
    // backward: given upstream gradient dL/dC, compute dL/dA and dL/dB
    //
    // WHY these formulas:
    //   forward is C = A @ B, so by the chain rule:
    //     dL/dA = dL/dC @ B^T     shape: [M×N] @ [N×K] → [M×K] ✓
    //     dL/dB = A^T @ dL/dC     shape: [K×M] @ [M×N] → [K×N] ✓
    //
    // WHY transpose() is cheap here:
    //   transpose() just swaps stride values — no data is copied.
    //   The forward() above handles strided inputs natively, so these
    //   matmul calls use the same fast path as the forward pass.
    //
    // WHY grads is size 2 with conditional fill:
    //   If a tensor didn't require grad (e.g. frozen weights) we skip
    //   computing its gradient entirely — no wasted matmul.
    //   The autograd engine reads grads[0] for A and grads[1] for B.
    // ----------------------------------------------------------------
    static std::vector<Tensor> backward(
        const Tensor& grad,
        const Tensor& saved_A, const Tensor& saved_B,
        bool rA, bool rB)
    {
        std::vector<Tensor> grads(2);
        if (rA) grads[0] = matmul(grad, saved_B.transpose());  // dL/dA
        if (rB) grads[1] = matmul(saved_A.transpose(), grad);  // dL/dB
        return grads;
    }
};

// ----------------------------------------------------------------
// matmul: public entry point — runs forward, wires autograd graph
//
// WHY capture implA/implB instead of the full Tensor:
//   A Tensor holds both implementation (data) and autograd_meta (graph node).
//   If we captured the full Tensor in the lambda, the lambda would hold a
//   reference to autograd_meta, which itself holds other lambdas, which hold
//   other tensors — forming a reference cycle that leaks the entire compute
//   graph. Capturing only the implementation (raw data + shape) breaks the
//   cycle: backward can still read the saved weights but doesn't keep the
//   graph alive.
//
// WHY NoGradGuard inside the autograd block:
//   backward() calls matmul() recursively. Without the guard those inner
//   calls would try to build autograd nodes for the gradient computation,
//   which is unnecessary and would grow the graph unboundedly.
//   NoGradGuard sets grad_mode_enabled=false for its scope, so the recursive
//   matmul calls produce plain tensors with no graph attachment.
// ----------------------------------------------------------------
inline Tensor matmul(const Tensor& A, const Tensor& B) {
    assert(A.ndim() == 2 && B.ndim() == 2);
    assert(A.shape(1) == B.shape(0));   // inner dimensions must match

    Tensor C = MatMulOp::forward(A, B);

    bool rA = A.requires_grad(), rB = B.requires_grad();
    if (grad_mode_enabled && (rA || rB)) {
        NoGradGuard no_grad;

        // capture data pointers only — avoids reference cycle (see above)
        auto implA = A.implementation;
        auto implB = B.implementation;

        C.autograd_meta = make_grad_meta(
            "matmul",
            {A.autograd_meta, B.autograd_meta},
            [implA, implB, rA, rB](const Tensor& grad) {
                // reconstruct bare tensors with data but no graph node —
                // these are read-only inputs to the gradient matmuls
                Tensor sA; sA.implementation = implA;
                Tensor sB; sB.implementation = implB;
                return MatMulOp::backward(grad, sA, sB, rA, rB);
            });
    }
    return C;
}