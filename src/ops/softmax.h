#pragma once
#include <cmath>
#include "tensorlib.h"
#include "ops/ops.h"


inline Tensor softmax(const Tensor& x, int dim = 0) {
    assert(x.ndim() == 2);
    assert(dim == 0 || dim == 1);

    int64_t rows = x.shape(0);
    int64_t cols = x.shape(1);

    // n_slices: how many independent vectors we normalise
    // slice_len: the length of each vector being normalised
    // dim=0: normalise across rows for each column independently
    // dim=1: normalise across cols for each row independently
    int64_t n_slices  = (dim == 0) ? cols : rows;
    int64_t slice_len = (dim == 0) ? rows : cols;

    Tensor out = Tensor::zeros({rows, cols});

    for (int64_t s = 0; s < n_slices; s++) {

        // ---- step 1: find max for numerical stability ----
        // subtracting max before exp prevents overflow
        // mathematically equivalent: max cancels in numerator and denominator
        // without this: exp(1000) = inf, exp(inf)/sum(inf) = nan
        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t i = 0; i < slice_len; i++) {
            float v = (dim == 0) ? x.at(i, s) : x.at(s, i);
            if (v > max_val) max_val = v;
        }

        // ---- step 2: compute exp(x - max) and accumulate sum ----
        float sum_exp = 0.f;
        for (int64_t i = 0; i < slice_len; i++) {
            float v  = (dim == 0) ? x.at(i, s) : x.at(s, i);
            float e  = std::exp(v - max_val);   // largest value becomes exp(0)=1
            sum_exp += e;
            if (dim == 0) out.at(i, s) = e;
            else          out.at(s, i) = e;
        }

        // ---- step 3: divide by sum → values now sum to 1 ----
        for (int64_t i = 0; i < slice_len; i++) {
            if (dim == 0) out.at(i, s) /= sum_exp;
            else          out.at(s, i) /= sum_exp;
        }
    }

    if (grad_mode_enabled && x.requires_grad()) {

        NoGradGuard no_grad;
        
        std::shared_ptr<AutogradMeta> autograd_meta_x = x.autograd_meta;

        // save output — backward only needs the softmax output values,
        // not the original input. this is more efficient than saving x
        // because we'd have to recompute softmax anyway
        Tensor saved_out = out.clone();

        std::function<std::vector<Tensor>(const Tensor&)> backward_fn =
            [saved_out, rows, cols, dim, n_slices, slice_len]
            (const Tensor& grad) -> std::vector<Tensor> {

                Tensor dx = Tensor::zeros({rows, cols});

                for (int64_t s = 0; s < n_slices; s++) {

                    // ---- step 1: dot = sum_i(grad_i * out_i) ----
                    //
                    // the full Jacobian of softmax is:
                    //   J_ij = s_i * (delta_ij - s_j)
                    //
                    // applying it to upstream grad:
                    //   dx_j = sum_i(grad_i * J_ij)
                    //        = sum_i(grad_i * s_i * delta_ij)
                    //          - sum_i(grad_i * s_i * s_j)
                    //        = grad_j * s_j - s_j * sum_i(grad_i * s_i)
                    //        = s_j * (grad_j - dot)
                    //
                    // where dot = sum_i(grad_i * s_i)
                    // computing dot first avoids materialising the full Jacobian matrix
                    float dot = 0.f;
                    for (int64_t i = 0; i < slice_len; i++) {
                        float g = (dim == 0) ? grad.at(i, s)      : grad.at(s, i);
                        float o = (dim == 0) ? saved_out.at(i, s) : saved_out.at(s, i);
                        dot += g * o;
                    }

                    // ---- step 2: dx_j = s_j * (grad_j - dot) ----
                    //
                    // the dot subtraction enforces that dx sums to zero —
                    // required because softmax outputs sum to 1, so their
                    // gradients must sum to 0 (perturbations must be zero-sum)
                    for (int64_t i = 0; i < slice_len; i++) {
                        float g = (dim == 0) ? grad.at(i, s)      : grad.at(s, i);
                        float o = (dim == 0) ? saved_out.at(i, s) : saved_out.at(s, i);

                        float d = o * (g - dot);

                        if (dim == 0) dx.at(i, s) = d;
                        else          dx.at(s, i) = d;
                    }
                }

                return {dx};
            };

        out.autograd_meta = make_grad_meta(
            "softmax",
            {autograd_meta_x},
            backward_fn);
    }

    return out;
}