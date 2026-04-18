#include "ops/im2col.h"

#include <cassert>
#include <omp.h>

Tensor Im2ColOp::forward(const Tensor& x,
                         size_t kernel_h,
                         size_t kernel_w,
                         size_t stride_h,
                         size_t stride_w,
                         size_t pad_h,
                         size_t pad_w) {
    assert(x.ndim == 4);
    assert(kernel_h > 0 && kernel_w > 0);
    assert(stride_h > 0 && stride_w > 0);

    const size_t N = x.shape[0];
    const size_t C = x.shape[1];
    const size_t H = x.shape[2];
    const size_t W = x.shape[3];

    assert(H + 2 * pad_h >= kernel_h);
    assert(W + 2 * pad_w >= kernel_w);
    assert((H + 2 * pad_h - kernel_h) % stride_h == 0);
    assert((W + 2 * pad_w - kernel_w) % stride_w == 0);

    const size_t out_h = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    const size_t out_w = (W + 2 * pad_w - kernel_w) / stride_w + 1;

    Shape out_shape{};
    out_shape[0] = N * out_h * out_w;
    out_shape[1] = C * kernel_h * kernel_w;
    Tensor cols = Tensor::zeros(out_shape, 2);

    float* col_ptr = cols.storage->data;
    const float* x_ptr = x.storage->data;
    const size_t sN = x.strides[0];
    const size_t sC = x.strides[1];
    const size_t sH = x.strides[2];
    const size_t sW = x.strides[3];

    const size_t K = out_shape[1];

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t n = 0; n < N; ++n) {
        for (size_t oh = 0; oh < out_h; ++oh) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                const size_t row = (n * out_h + oh) * out_w + ow;
                float* dst = col_ptr + row * K;

                size_t k = 0;
                for (size_t c = 0; c < C; ++c) {
                    for (size_t kh = 0; kh < kernel_h; ++kh) {
                        const int ih = static_cast<int>(oh * stride_h + kh) - static_cast<int>(pad_h);
                        for (size_t kw = 0; kw < kernel_w; ++kw) {
                            const int iw = static_cast<int>(ow * stride_w + kw) - static_cast<int>(pad_w);

                            if (ih >= 0 && iw >= 0 &&
                                static_cast<size_t>(ih) < H &&
                                static_cast<size_t>(iw) < W) {
                                const size_t x_idx = x.offset +
                                                     n * sN +
                                                     c * sC +
                                                     static_cast<size_t>(ih) * sH +
                                                     static_cast<size_t>(iw) * sW;
                                dst[k++] = x_ptr[x_idx];
                            } else {
                                dst[k++] = 0.0f;
                            }
                        }
                    }
                }
            }
        }
    }

    return cols;
}

Tensor im2col(const Tensor& x,
              size_t kernel_h,
              size_t kernel_w,
              size_t stride_h,
              size_t stride_w,
              size_t pad_h,
              size_t pad_w) {
    return Im2ColOp::forward(x, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
}
