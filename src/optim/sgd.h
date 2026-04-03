#pragma once

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include "tensor/tensor.h"

class SGD {
public:
    SGD(std::vector<Tensor*> params, float learning_rate)
        : params(std::move(params)), learning_rate(learning_rate) {
        assert(learning_rate > 0.0f);
    }

    void zero_grad() {
        for (Tensor* p : params) {
            if (p && p->autograd_meta)
                p->autograd_meta->grad = nullptr;
        }
    }

    void step() {
        for (Tensor* p : params) {
            if (!p || !p->autograd_meta || !p->autograd_meta->grad)
                continue;

            Tensor& g = *p->autograd_meta->grad;
            assert(p->ndim == g.ndim);
            for (size_t d = 0; d < p->ndim; ++d)
                assert(p->shape[d] == g.shape[d]);

            if (p->is_contiguous() && g.is_contiguous()) {
                float* dst = p->storage->data + p->offset;
                const float* src = g.storage->data + g.offset;
                for (size_t i = 0; i < p->numel; ++i)
                    dst[i] -= learning_rate * src[i];
                continue;
            }

            const Strides cs = Tensor::strides_from_shape(p->shape, p->ndim);
            for (size_t flat = 0; flat < p->numel; ++flat) {
                size_t p_idx = p->offset;
                size_t g_idx = g.offset;
                size_t rem = flat;

                for (size_t d = 0; d < p->ndim; ++d) {
                    const size_t dim_idx = rem / cs[d];
                    rem %= cs[d];
                    p_idx += dim_idx * p->strides[d];
                    g_idx += dim_idx * g.strides[d];
                }

                p->storage->data[p_idx] -= learning_rate * g.storage->data[g_idx];
            }
        }
    }

private:
    std::vector<Tensor*> params;
    float learning_rate;
};
