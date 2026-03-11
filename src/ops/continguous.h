#pragma once

#include "tensorlib.h"

struct ContiguousOp {

    // forward: copy elements into packed row-major storage
    // returns a new tensor with default strides
    static Tensor forward(const Tensor& t) {
        auto new_storage = std::make_shared<Storage>(t.numel());
        std::vector<int64_t> new_strides = default_strides(t.shape());

        auto new_impl = std::make_shared<TensorImpl>(
            new_storage,
            t.shape(),
            new_strides,
            0
        );

        const auto& shape   = t.shape();
        const auto& strides = t.strides();
        const size_t offset = t.implementation->offset;
        const float* src    = t.implementation->storage->ptr();
        float*       dst    = new_storage->ptr();
        int64_t ndim        = (int64_t)shape.size();
        int64_t numel       = t.numel();

        std::vector<int64_t> idx(ndim, 0);
        for (int64_t flat = 0; flat < numel; ++flat) {
            int64_t src_idx = offset;
            for (int64_t d = 0; d < ndim; ++d)
                src_idx += idx[d] * strides[d];
            dst[flat] = src[src_idx];

            // row-major carry
            for (int64_t d = ndim - 1; d >= 0; --d) {
                if (++idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }

        Tensor result;
        result.implementation = new_impl;
        return result;
    }

    // backward: copy is an identity operation element-wise
    // gradient passes straight through — values unchanged
    // the engine handles layout differences during accumulation
    static std::vector<Tensor> backward(const Tensor& grad) {
        return { grad.clone() };
    }
};

inline Tensor contiguous(const Tensor& tensor) {

    // fast path: already contiguous
    // return same storage, share autograd_meta — no graph node needed
    // because no data transformation happened
    if (tensor.is_contiguous()) {
        Tensor result;
        result.implementation = tensor.implementation;
        result.autograd_meta  = tensor.autograd_meta;
        return result;
    }

    // slow path: copy into packed storage via ContiguousOp
    Tensor out = ContiguousOp::forward(tensor);

    // wire into graph so gradient flows back through the copy
    if (grad_mode_enabled && tensor.requires_grad()) {
        
        NoGradGuard no_grad;

        out.autograd_meta = make_grad_meta(
            "contiguous",
            {tensor.autograd_meta},
            [](const Tensor& grad) {
                return ContiguousOp::backward(grad);
            });
    }

    return out;
}