#include <gtest/gtest.h>

#include <cstddef>
#include <initializer_list>

#include "ops/im2col.h"

namespace {

static Shape make_shape(std::initializer_list<size_t> dims) {
    Shape s{};
    size_t i = 0;
    for (size_t d : dims)
        s[i++] = d;
    return s;
}

static Tensor make_tensor(std::initializer_list<size_t> dims,
                          std::initializer_list<float> vals) {
    Shape s = make_shape(dims);
    const size_t ndim = dims.size();
    Tensor t(s, ndim);
    size_t i = 0;
    for (float v : vals)
        t.storage->data[i++] = v;
    return t;
}

}  // namespace

class Im2ColOpTest : public ::testing::Test {};

TEST_F(Im2ColOpTest, OutputShapeNoPaddingNoStride) {
    Tensor x = Tensor::zeros(make_shape({2, 3, 5, 5}), 4);
    Tensor cols = Im2ColOp::forward(x, /*kernel_h=*/3, /*kernel_w=*/3);

    // out_h=3, out_w=3 -> rows = N*out_h*out_w = 18, cols = C*kh*kw = 27
    EXPECT_EQ(cols.ndim, 2u);
    EXPECT_EQ(cols.shape_at(0), 18u);
    EXPECT_EQ(cols.shape_at(1), 27u);
}

TEST_F(Im2ColOpTest, SingleChannelValuesMatchReference) {
    Tensor x = make_tensor(
        {1, 1, 3, 3},
        {1.f, 2.f, 3.f,
         4.f, 5.f, 6.f,
         7.f, 8.f, 9.f}
    );

    Tensor cols = Im2ColOp::forward(x, /*kernel_h=*/2, /*kernel_w=*/2);

    ASSERT_EQ(cols.shape_at(0), 4u);
    ASSERT_EQ(cols.shape_at(1), 4u);

    // Row order: (oh,ow) = (0,0), (0,1), (1,0), (1,1)
    EXPECT_FLOAT_EQ(cols(0, 0), 1.f);
    EXPECT_FLOAT_EQ(cols(0, 1), 2.f);
    EXPECT_FLOAT_EQ(cols(0, 2), 4.f);
    EXPECT_FLOAT_EQ(cols(0, 3), 5.f);

    EXPECT_FLOAT_EQ(cols(1, 0), 2.f);
    EXPECT_FLOAT_EQ(cols(1, 1), 3.f);
    EXPECT_FLOAT_EQ(cols(1, 2), 5.f);
    EXPECT_FLOAT_EQ(cols(1, 3), 6.f);

    EXPECT_FLOAT_EQ(cols(2, 0), 4.f);
    EXPECT_FLOAT_EQ(cols(2, 1), 5.f);
    EXPECT_FLOAT_EQ(cols(2, 2), 7.f);
    EXPECT_FLOAT_EQ(cols(2, 3), 8.f);

    EXPECT_FLOAT_EQ(cols(3, 0), 5.f);
    EXPECT_FLOAT_EQ(cols(3, 1), 6.f);
    EXPECT_FLOAT_EQ(cols(3, 2), 8.f);
    EXPECT_FLOAT_EQ(cols(3, 3), 9.f);
}

TEST_F(Im2ColOpTest, PaddingInsertsZeros) {
    Tensor x = make_tensor(
        {1, 1, 2, 2},
        {1.f, 2.f,
         3.f, 4.f}
    );

    Tensor cols = Im2ColOp::forward(
        x,
        /*kernel_h=*/3,
        /*kernel_w=*/3,
        /*stride_h=*/1,
        /*stride_w=*/1,
        /*pad_h=*/1,
        /*pad_w=*/1
    );

    ASSERT_EQ(cols.shape_at(0), 4u);
    ASSERT_EQ(cols.shape_at(1), 9u);

    // Top-left patch with padding around original image.
    EXPECT_FLOAT_EQ(cols(0, 0), 0.f);
    EXPECT_FLOAT_EQ(cols(0, 1), 0.f);
    EXPECT_FLOAT_EQ(cols(0, 2), 0.f);
    EXPECT_FLOAT_EQ(cols(0, 3), 0.f);
    EXPECT_FLOAT_EQ(cols(0, 4), 1.f);
    EXPECT_FLOAT_EQ(cols(0, 5), 2.f);
    EXPECT_FLOAT_EQ(cols(0, 6), 0.f);
    EXPECT_FLOAT_EQ(cols(0, 7), 3.f);
    EXPECT_FLOAT_EQ(cols(0, 8), 4.f);
}

TEST_F(Im2ColOpTest, InvalidInputRankAsserts) {
    Tensor bad = Tensor::zeros(make_shape({1, 4, 4}), 3);
    EXPECT_DEATH(Im2ColOp::forward(bad, 3, 3), "");
}
