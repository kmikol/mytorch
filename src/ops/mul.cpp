#include "ops/mul.h"

#include <cassert>


Tensor MulOp::forward(const Tensor& a, const Tensor& b) {
    assert(a.ndim == b.ndim);
    for (size_t d = 0; d < a.ndim; ++d)
        assert(a.shape[d] == b.shape[d]);

    Tensor out(a.shape, a.ndim);
    for (size_t i = 0; i < a.numel; ++i)
        out.flat(i) = a.flat(i) * b.flat(i);

    return out;
}

std::vector<Tensor> MulOp::backward(const Tensor& grad,
                                    const Tensor& a,
                                    const Tensor& b) {
    Tensor grad_a(a.shape, a.ndim);
    Tensor grad_b(b.shape, b.ndim);
    for (size_t i = 0; i < grad.numel; ++i) {
        grad_a.flat(i) = grad.flat(i) * b.flat(i);
        grad_b.flat(i) = grad.flat(i) * a.flat(i);
    }
    return {grad_a, grad_b};
}

Tensor mul(const Tensor& a, const Tensor& b) {
    Tensor out = MulOp::forward(a, b);

    if (grad_mode_enabled && (a.requires_grad() || b.requires_grad())) {
        // Capture a and b by value: Tensor copies share the same
        // shared_ptr<Storage>, so this is a zero-copy view of each input.
        // Safe as long as no in-place mutations are made to a or b after
        // this call.
        out.autograd_meta = make_grad_meta(
            "MulOp",
            {a.autograd_meta, b.autograd_meta},
            [a_save = a, b_save = b](const Tensor& grad) {
                return MulOp::backward(grad, a_save, b_save);
            }
        );
    }

    return out;
}
