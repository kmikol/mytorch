#include <gtest/gtest.h>

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>

#include "autograd.h"
#include "optim/sgd.h"
#include "ops/matmul.h"

namespace {

static Shape make_shape(std::initializer_list<size_t> dims) {
    Shape s{};
    size_t i = 0;
    for (size_t d : dims)
        s[i++] = d;
    return s;
}

static Tensor make_tensor_2d(size_t rows,
                             size_t cols,
                             const std::vector<float>& values,
                             bool requires_grad = false) {
    EXPECT_EQ(values.size(), rows * cols);

    Tensor t(make_shape({rows, cols}), 2, requires_grad);
    size_t idx = 0;
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            t(i, j) = values[idx++];

    return t;
}

}  // namespace

class SGDTest : public ::testing::Test {};

TEST_F(SGDTest, StepAppliesLearningRateScaledGradient) {
    Tensor w = make_tensor_2d(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, /*requires_grad=*/true);
    Tensor b = make_tensor_2d(1, 2, {0.5f, -0.5f}, /*requires_grad=*/true);

    w.autograd_meta->grad = std::make_shared<Tensor>(
        make_tensor_2d(2, 2, {0.1f, -0.2f, 1.5f, 2.0f}));
    b.autograd_meta->grad = std::make_shared<Tensor>(
        make_tensor_2d(1, 2, {0.4f, -0.6f}));

    SGD opt({&w, &b}, /*learning_rate=*/0.5f);
    opt.step();

    EXPECT_FLOAT_EQ(w(0, 0), 1.0f - 0.5f * 0.1f);
    EXPECT_FLOAT_EQ(w(0, 1), 2.0f - 0.5f * (-0.2f));
    EXPECT_FLOAT_EQ(w(1, 0), 3.0f - 0.5f * 1.5f);
    EXPECT_FLOAT_EQ(w(1, 1), 4.0f - 0.5f * 2.0f);

    EXPECT_FLOAT_EQ(b(0, 0), 0.5f - 0.5f * 0.4f);
    EXPECT_FLOAT_EQ(b(0, 1), -0.5f - 0.5f * (-0.6f));
}

TEST_F(SGDTest, StepSkipsParamsWithoutGradient) {
    Tensor p_with_grad = make_tensor_2d(1, 2, {2.0f, 4.0f}, /*requires_grad=*/true);
    Tensor p_without_grad = make_tensor_2d(1, 2, {5.0f, 7.0f}, /*requires_grad=*/true);

    p_with_grad.autograd_meta->grad = std::make_shared<Tensor>(
        make_tensor_2d(1, 2, {1.0f, 3.0f}));

    SGD opt({&p_with_grad, &p_without_grad}, /*learning_rate=*/0.1f);
    opt.step();

    EXPECT_FLOAT_EQ(p_with_grad(0, 0), 1.9f);
    EXPECT_FLOAT_EQ(p_with_grad(0, 1), 3.7f);

    EXPECT_FLOAT_EQ(p_without_grad(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(p_without_grad(0, 1), 7.0f);
}

TEST_F(SGDTest, ZeroGradClearsAccumulatedGradientPointers) {
    Tensor w = make_tensor_2d(2, 1, {1.0f, 2.0f}, /*requires_grad=*/true);
    Tensor b = make_tensor_2d(1, 1, {0.0f}, /*requires_grad=*/true);

    w.autograd_meta->grad = std::make_shared<Tensor>(make_tensor_2d(2, 1, {3.0f, 4.0f}));
    b.autograd_meta->grad = std::make_shared<Tensor>(make_tensor_2d(1, 1, {5.0f}));

    SGD opt({&w, &b}, /*learning_rate=*/0.01f);
    opt.zero_grad();

    EXPECT_FALSE(w.has_grad());
    EXPECT_FALSE(b.has_grad());
}

TEST_F(SGDTest, StepAndZeroGradWorkWithRealBackwardGradients) {
    Tensor x = make_tensor_2d(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, /*requires_grad=*/true);
    Tensor w = make_tensor_2d(2, 2, {1.0f, 0.0f, 0.0f, 1.0f}, /*requires_grad=*/true);

    Tensor y = matmul(x, w);
    backward(y);

    ASSERT_TRUE(w.has_grad());
    const float g00 = w.grad()(0, 0);
    const float g01 = w.grad()(0, 1);
    const float g10 = w.grad()(1, 0);
    const float g11 = w.grad()(1, 1);

    SGD opt({&w}, /*learning_rate=*/0.25f);
    opt.step();

    EXPECT_FLOAT_EQ(w(0, 0), 1.0f - 0.25f * g00);
    EXPECT_FLOAT_EQ(w(0, 1), 0.0f - 0.25f * g01);
    EXPECT_FLOAT_EQ(w(1, 0), 0.0f - 0.25f * g10);
    EXPECT_FLOAT_EQ(w(1, 1), 1.0f - 0.25f * g11);

    opt.zero_grad();
    EXPECT_FALSE(w.has_grad());
}

TEST_F(SGDTest, StepUpdatesFourDimensionalParameters) {
    Tensor w4 = Tensor::ones(make_shape({2, 1, 2, 2}), 4, /*requires_grad=*/true);
    Tensor g4 = Tensor::ones(make_shape({2, 1, 2, 2}), 4);
    w4.autograd_meta->grad = std::make_shared<Tensor>(g4);

    SGD opt({&w4}, /*learning_rate=*/0.25f);
    opt.step();

    for (size_t oc = 0; oc < 2; ++oc)
        for (size_t ic = 0; ic < 1; ++ic)
            for (size_t kh = 0; kh < 2; ++kh)
                for (size_t kw = 0; kw < 2; ++kw)
                    EXPECT_FLOAT_EQ(w4(oc, ic, kh, kw), 0.75f);
}
