#include "layers/conv2d.h"

#include <cassert>
#include <cmath>
#include <random>

#include "ops/conv2d.h"

namespace {

static Shape make_shape_2d(size_t d0, size_t d1) {
    Shape s{};
    s[0] = d0;
    s[1] = d1;
    return s;
}

static Shape make_shape_4d(size_t d0, size_t d1, size_t d2, size_t d3) {
    Shape s{};
    s[0] = d0;
    s[1] = d1;
    s[2] = d2;
    s[3] = d3;
    return s;
}

}  // namespace

Conv2d::Conv2d(size_t in_channels,
               size_t out_channels,
               size_t kernel_h,
               size_t kernel_w,
               size_t stride_h,
               size_t stride_w,
               size_t padding_h,
               size_t padding_w)
    : in_channels(in_channels),
      out_channels(out_channels),
      kernel_h(kernel_h),
      kernel_w(kernel_w),
      stride_h(stride_h),
      stride_w(stride_w),
      padding_h(padding_h),
      padding_w(padding_w),
      weight(make_shape_4d(out_channels, in_channels, kernel_h, kernel_w), 4, true),
      bias(Tensor::zeros(make_shape_2d(1, out_channels), 2, true)) {
    assert(in_channels > 0);
    assert(out_channels > 0);
    assert(kernel_h > 0 && kernel_w > 0);
    assert(stride_h > 0 && stride_w > 0);

    const float fan_in = static_cast<float>(in_channels * kernel_h * kernel_w);
    const float fan_out = static_cast<float>(out_channels * kernel_h * kernel_w);
    const float limit = std::sqrt(6.0f / (fan_in + fan_out));

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (size_t oc = 0; oc < out_channels; ++oc) {
        for (size_t ic = 0; ic < in_channels; ++ic) {
            for (size_t kh = 0; kh < kernel_h; ++kh) {
                for (size_t kw = 0; kw < kernel_w; ++kw)
                    weight(oc, ic, kh, kw) = dist(rng);
            }
        }
    }
}

Tensor Conv2d::forward(const Tensor& x) const {
    assert(x.ndim == 4);
    assert(x.shape[1] == in_channels);
    return conv2d(x, weight, bias, stride_h, stride_w, padding_h, padding_w);
}

std::vector<Tensor*> Conv2d::parameters() {
    return {&weight, &bias};
}
