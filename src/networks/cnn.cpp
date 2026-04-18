#include "networks/cnn.h"

#include <cassert>
#include <utility>

#include "ops/reshape.h"

namespace {

static size_t conv_out_dim(size_t in, size_t kernel, size_t stride, size_t pad) {
    assert(in + 2 * pad >= kernel);
    assert((in + 2 * pad - kernel) % stride == 0);
    return (in + 2 * pad - kernel) / stride + 1;
}

static Shape make_shape_2d(size_t d0, size_t d1) {
    Shape s{};
    s[0] = d0;
    s[1] = d1;
    return s;
}

}  // namespace

CNN::CNN(size_t input_channels,
         size_t input_height,
         size_t input_width,
         size_t conv_out_channels,
         size_t kernel_h,
         size_t kernel_w,
         ActivationFn activation,
         size_t output_features,
         size_t stride_h,
         size_t stride_w,
         size_t padding_h,
         size_t padding_w)
    : conv1_(
        input_channels,
        conv_out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w
      ),
      classifier_(
        conv_out_channels *
            conv_out_dim(input_height, kernel_h, stride_h, padding_h) *
            conv_out_dim(input_width, kernel_w, stride_w, padding_w),
        output_features
      ),
      activation_(std::move(activation)),
      flattened_features_(
        conv_out_channels *
            conv_out_dim(input_height, kernel_h, stride_h, padding_h) *
            conv_out_dim(input_width, kernel_w, stride_w, padding_w)
      ) {
    assert(input_channels > 0);
    assert(output_features > 0);
    assert(flattened_features_ > 0);
    assert(static_cast<bool>(activation_));
}

Tensor CNN::forward(const Tensor& x) const {
    assert(x.ndim == 4);
    Tensor h = activation_(conv1_.forward(x));

    const size_t batch_size = h.shape[0];
    Tensor flat = reshape(h, make_shape_2d(batch_size, flattened_features_), 2);
    return classifier_.forward(flat);
}

std::vector<Tensor*> CNN::parameters() {
    std::vector<Tensor*> params = conv1_.parameters();
    std::vector<Tensor*> cls_params = classifier_.parameters();
    params.insert(params.end(), cls_params.begin(), cls_params.end());
    return params;
}
