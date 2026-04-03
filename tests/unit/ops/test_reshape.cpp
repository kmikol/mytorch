#include <gtest/gtest.h>

#include <cstddef>
#include <initializer_list>

#include "autograd.h"
#include "ops/reshape.h"

namespace {

static Shape make_shape(std::initializer_list<size_t> dims) {
    Shape s{};
    size_t i = 0;
    for (size_t d : dims)
        s[i++] = d;
    return s;
}

}  // namespace

class ReshapeOpTest : public ::testing::Test {};

TEST_F(ReshapeOpTest, ForwardChangesShapeWithoutCopying) {
    Tensor x = Tensor::zeros(make_shape({2, 3}), 2);
    for (size_t i = 0; i < 6; ++i)
        x.storage->data[i] = static_cast<float>(i + 1);

    Tensor y = reshape(x, make_shape({1, 6}), 2);
    EXPECT_EQ(y.shape_at(0), 1u);
    EXPECT_EQ(y.shape_at(1), 6u);
    EXPECT_EQ(y.storage.get(), x.storage.get());
    EXPECT_FLOAT_EQ(y(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(y(0, 5), 6.0f);
}

TEST_F(ReshapeOpTest, BackwardPropagatesGradientToOriginalShape) {
    Tensor x = Tensor::ones(make_shape({2, 3}), 2, /*requires_grad=*/true);
    Tensor y = reshape(x, make_shape({1, 6}), 2);

    backward(y);

    ASSERT_TRUE(x.has_grad());
    EXPECT_EQ(x.grad().shape_at(0), 2u);
    EXPECT_EQ(x.grad().shape_at(1), 3u);

    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(x.grad()(i, j), 1.0f);
}
