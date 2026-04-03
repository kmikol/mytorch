#include "ops/conv2d.h"

#include <cassert>
#include <omp.h>

#include "ops/im2col.h"
#include "ops/matmul.h"

namespace {

static Tensor flatten_weight(const Tensor& weight) {
    assert(weight.ndim == 4);

    const size_t out_channels = weight.shape[0];
    const size_t in_channels = weight.shape[1];
    const size_t kernel_h = weight.shape[2];
    const size_t kernel_w = weight.shape[3];
    const size_t K = in_channels * kernel_h * kernel_w;

    Shape shape{};
    shape[0] = K;
    shape[1] = out_channels;
    Tensor flat(shape, 2);

    for (size_t oc = 0; oc < out_channels; ++oc) {
        for (size_t ic = 0; ic < in_channels; ++ic) {
            for (size_t kh = 0; kh < kernel_h; ++kh) {
                for (size_t kw = 0; kw < kernel_w; ++kw) {
                    const size_t k = (ic * kernel_h + kh) * kernel_w + kw;
                    flat(k, oc) = weight(oc, ic, kh, kw);
                }
            }
        }
    }

    return flat;
}

static Tensor unflatten_weight_grad(const Tensor& grad_flat,
                                    size_t out_channels,
                                    size_t in_channels,
                                    size_t kernel_h,
                                    size_t kernel_w) {
    Shape w_shape{};
    w_shape[0] = out_channels;
    w_shape[1] = in_channels;
    w_shape[2] = kernel_h;
    w_shape[3] = kernel_w;
    Tensor grad_w(w_shape, 4);

    for (size_t oc = 0; oc < out_channels; ++oc) {
        for (size_t ic = 0; ic < in_channels; ++ic) {
            for (size_t kh = 0; kh < kernel_h; ++kh) {
                for (size_t kw = 0; kw < kernel_w; ++kw) {
                    const size_t k = (ic * kernel_h + kh) * kernel_w + kw;
                    grad_w(oc, ic, kh, kw) = grad_flat(k, oc);
                }
            }
        }
    }

    return grad_w;
}

static Tensor flatten_output_grad(const Tensor& grad) {
    assert(grad.ndim == 4);

    const size_t N = grad.shape[0];
    const size_t C_out = grad.shape[1];
    const size_t out_h = grad.shape[2];
    const size_t out_w = grad.shape[3];

    Shape shape{};
    shape[0] = N * out_h * out_w;
    shape[1] = C_out;
    Tensor grad_2d(shape, 2);

    for (size_t n = 0; n < N; ++n) {
        for (size_t oh = 0; oh < out_h; ++oh) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                const size_t row = (n * out_h + oh) * out_w + ow;
                for (size_t oc = 0; oc < C_out; ++oc)
                    grad_2d(row, oc) = grad(n, oc, oh, ow);
            }
        }
    }

    return grad_2d;
}

static Tensor col2im(const Tensor& grad_cols,
                     size_t N,
                     size_t C,
                     size_t H,
                     size_t W,
                     size_t kernel_h,
                     size_t kernel_w,
                     size_t stride_h,
                     size_t stride_w,
                     size_t pad_h,
                     size_t pad_w) {
    const size_t out_h = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    const size_t out_w = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    Shape shape{};
    shape[0] = N;
    shape[1] = C;
    shape[2] = H;
    shape[3] = W;
    Tensor grad_x = Tensor::zeros(shape, 4);

    const float* col_ptr = grad_cols.storage->data;
    float* gx = grad_x.storage->data;
    const size_t K = C * kernel_h * kernel_w;

    #pragma omp parallel for schedule(static)
    for (size_t n = 0; n < N; ++n) {
        for (size_t oh = 0; oh < out_h; ++oh) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                const size_t row = (n * out_h + oh) * out_w + ow;
                const float* src = col_ptr + row * K;

                size_t k = 0;
                for (size_t c = 0; c < C; ++c) {
                    for (size_t kh = 0; kh < kernel_h; ++kh) {
                        const int ih = static_cast<int>(oh * stride_h + kh) - static_cast<int>(pad_h);
                        for (size_t kw = 0; kw < kernel_w; ++kw) {
                            const int iw = static_cast<int>(ow * stride_w + kw) - static_cast<int>(pad_w);
                            if (ih >= 0 && iw >= 0 &&
                                static_cast<size_t>(ih) < H &&
                                static_cast<size_t>(iw) < W) {
                                const size_t idx = ((n * C + c) * H + static_cast<size_t>(ih)) * W + static_cast<size_t>(iw);
                                gx[idx] += src[k];
                            }
                            ++k;
                        }
                    }
                }
            }
        }
    }

    return grad_x;
}

}  // namespace

Tensor Conv2dOp::forward(const Tensor& x,
                         const Tensor& weight,
                         const Tensor& bias,
                         size_t stride_h,
                         size_t stride_w,
                         size_t pad_h,
                         size_t pad_w) {
    assert(x.ndim == 4);
    assert(weight.ndim == 4);
    assert(bias.ndim == 2);
    assert(stride_h > 0 && stride_w > 0);

    const size_t N = x.shape[0];
    const size_t in_channels = x.shape[1];
    const size_t H = x.shape[2];
    const size_t W = x.shape[3];

    const size_t out_channels = weight.shape[0];
    assert(weight.shape[1] == in_channels);
    const size_t kernel_h = weight.shape[2];
    const size_t kernel_w = weight.shape[3];

    assert(bias.shape[0] == 1);
    assert(bias.shape[1] == out_channels);

    assert(H + 2 * pad_h >= kernel_h);
    assert(W + 2 * pad_w >= kernel_w);
    assert((H + 2 * pad_h - kernel_h) % stride_h == 0);
    assert((W + 2 * pad_w - kernel_w) % stride_w == 0);

    const size_t out_h = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    const size_t out_w = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    Tensor cols = Im2ColOp::forward(x, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    Tensor weight_2d = flatten_weight(weight);
    Tensor out_2d = MatMulOp::forward(cols, weight_2d);

    for (size_t i = 0; i < out_2d.shape[0]; ++i) {
        for (size_t oc = 0; oc < out_channels; ++oc)
            out_2d(i, oc) += bias(0, oc);
    }

    Shape out_shape{};
    out_shape[0] = N;
    out_shape[1] = out_channels;
    out_shape[2] = out_h;
    out_shape[3] = out_w;
    Tensor out(out_shape, 4);

    for (size_t n = 0; n < N; ++n) {
        for (size_t oh = 0; oh < out_h; ++oh) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                const size_t row = (n * out_h + oh) * out_w + ow;
                for (size_t oc = 0; oc < out_channels; ++oc)
                    out(n, oc, oh, ow) = out_2d(row, oc);
            }
        }
    }

    return out;
}

std::vector<Tensor> Conv2dOp::backward(const Tensor& grad,
                                       const Tensor& x,
                                       const Tensor& weight,
                                       const Tensor& bias,
                                       size_t stride_h,
                                       size_t stride_w,
                                       size_t pad_h,
                                       size_t pad_w) {
    assert(grad.ndim == 4);
    (void)bias;

    const size_t N = x.shape[0];
    const size_t in_channels = x.shape[1];
    const size_t H = x.shape[2];
    const size_t W = x.shape[3];

    const size_t out_channels = weight.shape[0];
    const size_t kernel_h = weight.shape[2];
    const size_t kernel_w = weight.shape[3];

    Tensor cols = Im2ColOp::forward(x, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    Tensor grad_2d = flatten_output_grad(grad);

    Tensor grad_weight_2d = MatMulOp::forward(cols.T(), grad_2d);
    Tensor grad_weight = unflatten_weight_grad(
        grad_weight_2d,
        out_channels,
        in_channels,
        kernel_h,
        kernel_w
    );

    Shape grad_b_shape{};
    grad_b_shape[0] = 1;
    grad_b_shape[1] = out_channels;
    Tensor grad_bias = Tensor::zeros(grad_b_shape, 2);
    for (size_t i = 0; i < grad_2d.shape[0]; ++i) {
        for (size_t oc = 0; oc < out_channels; ++oc)
            grad_bias(0, oc) += grad_2d(i, oc);
    }

    Tensor weight_2d = flatten_weight(weight);
    Tensor grad_cols = MatMulOp::forward(grad_2d, weight_2d.T());
    Tensor grad_x = col2im(
        grad_cols,
        N,
        in_channels,
        H,
        W,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w
    );

    return {grad_x, grad_weight, grad_bias};
}

Tensor conv2d(const Tensor& x,
              const Tensor& weight,
              const Tensor& bias,
              size_t stride_h,
              size_t stride_w,
              size_t pad_h,
              size_t pad_w) {
    Tensor out = Conv2dOp::forward(x, weight, bias, stride_h, stride_w, pad_h, pad_w);

    const bool rx = x.requires_grad();
    const bool rw = weight.requires_grad();
    const bool rb = bias.requires_grad();

    if (grad_mode_enabled && (rx || rw || rb)) {
        std::vector<std::shared_ptr<AutogradMeta>> active_metas;
        if (rx) active_metas.push_back(x.autograd_meta);
        if (rw) active_metas.push_back(weight.autograd_meta);
        if (rb) active_metas.push_back(bias.autograd_meta);

        out.autograd_meta = make_grad_meta(
            "conv2d",
            std::move(active_metas),
            [x_save = x,
             w_save = weight,
             b_save = bias,
             rx,
             rw,
             rb,
             stride_h,
             stride_w,
             pad_h,
             pad_w](const Tensor& grad) {
                std::vector<Tensor> all_grads = Conv2dOp::backward(
                    grad,
                    x_save,
                    w_save,
                    b_save,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w
                );

                std::vector<Tensor> selected;
                if (rx) selected.push_back(all_grads[0]);
                if (rw) selected.push_back(all_grads[1]);
                if (rb) selected.push_back(all_grads[2]);
                return selected;
            }
        );
    }

    return out;
}
