#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>

#include "tensorlib.h"


struct SGD {

    // pointers to the weight tensors we are optimising
    std::vector<Tensor*> params;
    float lr;

    SGD(std::vector<Tensor*> params, float lr)
        : params(params), lr(lr) {}

    // set all gradients to null so they don't accumulate across steps
    void zero_grad() {
        for (Tensor* p : params) {
            if (p->autograd_meta != nullptr) {
                p->autograd_meta->grad = nullptr;
            }
        }
    }

    // nudge each weight in the direction that reduces the loss
    void step() {
        for (Tensor* p : params) {

            // skip if this param has no gradient
            if (p->autograd_meta == nullptr) continue;
            if (p->autograd_meta->grad == nullptr) continue;

            Tensor& grad = *p->autograd_meta->grad;

            for (int64_t r = 0; r < p->shape(0); r++) {
                for (int64_t c = 0; c < p->shape(1); c++) {
                    // move weight opposite to gradient direction
                    p->at(r, c) -= lr * grad.at(r, c);
                }
            }
        }
    }
};