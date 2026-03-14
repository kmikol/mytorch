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
            assert(p->ndim == 2 && g.ndim == 2);
            assert(p->shape[0] == g.shape[0] && p->shape[1] == g.shape[1]);

            for (size_t i = 0; i < p->shape[0]; ++i)
                for (size_t j = 0; j < p->shape[1]; ++j)
                    (*p)(i, j) -= learning_rate * g(i, j);
        }
    }

private:
    std::vector<Tensor*> params;
    float learning_rate;
};
