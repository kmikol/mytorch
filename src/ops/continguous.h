#pragma once

#include "tensorlib.h"

struct ContiguousOp {

    // forward: copy elements into packed row-major storage
    // returns a new tensor with default strides
    static Tensor forward(const Tensor& t) {
        auto new_storage = std::make_shared<Storage>(t.numel());
        std::vector<int64_t> new_strides = default_strides(t.implementation->shape);

        auto new_impl = std::make_shared<TensorImpl>(
            new_storage,
            t.implementation->shape,
            new_strides,
            0
        );

        // walk every element in logical order and copy into packed storage
        // for_each_index guarantees row-major traversal regardless of
        // the original tensor's strides
        int64_t flat = 0;
        for_each_index(t.implementation->shape, [&](const std::vector<int64_t>& idx) {
            new_storage->ptr()[flat++] = t.implementation->at(idx);
        });

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
    if (tensor.requires_grad()) {
        out.autograd_meta = make_grad_meta(
            "contiguous",
            {tensor.autograd_meta},
            [](const Tensor& grad) {
                return ContiguousOp::backward(grad);
            });
    }

    return out;
}