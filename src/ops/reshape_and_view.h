#pragma once
#include <vector>
#include <memory>
#include "tensorlib.h"

struct ViewOp {

    // forward: reinterpret the flat storage with a new shape
    // REQUIRES contiguous input — asserts otherwise
    static Tensor forward(const Tensor& t, const std::vector<int64_t>& new_shape) {
        assert(t.is_contiguous() &&
               "view requires contiguous tensor — use reshape() instead");

        int64_t new_numel = 1;
        for (int64_t s : new_shape) new_numel *= s;
        assert(new_numel == t.numel() &&
               "view: element count must be unchanged");

        Tensor result;
        result.implementation = std::make_shared<TensorImpl>(
            t.implementation->storage,     // same storage — zero copy
            new_shape,
            default_strides(new_shape),    // new strides for new shape
            t.implementation->offset
        );
        return result;
    }

    // backward: reshape the gradient back to the original shape
    // this is just another view — no data moves
    // original_shape is captured at forward time
    static std::vector<Tensor> backward(
        const Tensor& grad,
        const std::vector<int64_t>& original_shape)
    {
        // grad has the shape of the view output
        // we need to give back a gradient shaped like the view input
        // since view is contiguous by definition, this is always safe
        return { ViewOp::forward(grad, original_shape) };
    }
};

inline Tensor view(const Tensor& tensor, std::vector<int64_t> new_shape) {
    Tensor out = ViewOp::forward(tensor, new_shape);

    if (grad_mode_enabled && tensor.requires_grad()) {

        NoGradGuard no_grad;
        
        // capture the original shape so backward can restore it
        std::vector<int64_t> original_shape = tensor.implementation->shape;

        out.autograd_meta = make_grad_meta(
            "view",
            {tensor.autograd_meta},
            [original_shape](const Tensor& grad) {
                return ViewOp::backward(grad, original_shape);
            });
    }

    return out;
}

struct ReshapeOp {

    // reshape has no math of its own
    // it just decides whether a copy is needed before viewing
    static Tensor forward(const Tensor& t, const std::vector<int64_t>& new_shape) {
        int64_t new_numel = 1;
        for (int64_t s : new_shape) new_numel *= s;
        assert(new_numel == t.numel() &&
               "reshape: element count must be unchanged");

        if (t.is_contiguous()) {
            // fast path: already packed — view directly
            return t.view(new_shape);
        }

        // slow path: pack first, then view
        // note: we call the member functions here, not the static Op functions
        // so that gradient nodes are registered for each step
        return t.contiguous().view(new_shape);
    }

    // reshape has no backward of its own — the backward is handled
    // by whichever combination of view/contiguous nodes were registered
    // during forward. this is automatic because view() and contiguous()
    // both register themselves when requires_grad is true.
};

inline Tensor reshape(const Tensor& tensor, std::vector<int64_t> new_shape) {
    // reshape delegates entirely to ReshapeOp
    // gradient graph is built inside view() and contiguous()
    return ReshapeOp::forward(tensor, new_shape);
}