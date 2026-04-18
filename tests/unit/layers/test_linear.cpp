#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <vector>

#include "autograd.h"
#include "layers/linear.h"

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

class LinearCtorTest : public ::testing::Test {};

TEST_F(LinearCtorTest, ParameterShapesAndGradFlags) {
    Linear layer(/*in_features=*/3, /*out_features=*/2);

    EXPECT_EQ(layer.weight.ndim, 2u);
    EXPECT_EQ(layer.weight.shape_at(0), 3u);
    EXPECT_EQ(layer.weight.shape_at(1), 2u);

    EXPECT_EQ(layer.bias.ndim, 2u);
    EXPECT_EQ(layer.bias.shape_at(0), 1u);
    EXPECT_EQ(layer.bias.shape_at(1), 2u);

    EXPECT_TRUE(layer.weight.requires_grad());
    EXPECT_TRUE(layer.bias.requires_grad());
}

TEST_F(LinearCtorTest, BiasStartsAtZero) {
    Linear layer(/*in_features=*/5, /*out_features=*/4);

    for (size_t j = 0; j < layer.out_features; ++j)
        EXPECT_FLOAT_EQ(layer.bias(0, j), 0.0f);
}

TEST_F(LinearCtorTest, WeightInitWithinXavierUniformRange) {
    const size_t in_f = 6;
    const size_t out_f = 4;
    Linear layer(in_f, out_f);

    const float limit = std::sqrt(6.0f / static_cast<float>(in_f + out_f));
    for (size_t i = 0; i < in_f; ++i) {
        for (size_t j = 0; j < out_f; ++j) {
            const float w = layer.weight(i, j);
            EXPECT_LE(w, limit);
            EXPECT_GE(w, -limit);
        }
    }
}

TEST_F(LinearCtorTest, DistinctLayersUsuallyHaveDifferentWeights) {
    Linear a(/*in_features=*/8, /*out_features=*/8);
    Linear b(/*in_features=*/8, /*out_features=*/8);

    bool found_diff = false;
    for (size_t i = 0; i < 8 && !found_diff; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            if (a.weight(i, j) != b.weight(i, j)) {
                found_diff = true;
                break;
            }
        }
    }

    EXPECT_TRUE(found_diff);
}

TEST_F(LinearCtorTest, ParametersReturnsWeightThenBias) {
    Linear layer(/*in_features=*/3, /*out_features=*/2);
    auto params = layer.parameters();

    ASSERT_EQ(params.size(), 2u);
    EXPECT_EQ(params[0], &layer.weight);
    EXPECT_EQ(params[1], &layer.bias);
}

class LinearForwardTest : public ::testing::Test {};

TEST_F(LinearForwardTest, OutputShapeIsBatchByOutFeatures) {
    Linear layer(/*in_features=*/3, /*out_features=*/2);
    Tensor x = make_tensor_2d(
        4, 3,
        {1.0f, 2.0f, 3.0f,
         4.0f, 5.0f, 6.0f,
         7.0f, 8.0f, 9.0f,
         0.0f, 1.0f, 2.0f});

    Tensor y = layer.forward(x);
    EXPECT_EQ(y.ndim, 2u);
    EXPECT_EQ(y.shape_at(0), 4u);
    EXPECT_EQ(y.shape_at(1), 2u);
}

TEST_F(LinearForwardTest, AffineValuesMatchReference) {
    Linear layer(/*in_features=*/3, /*out_features=*/2);

    layer.weight = make_tensor_2d(
        3, 2,
        {1.0f, 2.0f,
         3.0f, 4.0f,
         5.0f, 6.0f},
        /*requires_grad=*/true);
    layer.bias = make_tensor_2d(1, 2, {10.0f, 20.0f}, /*requires_grad=*/true);

    Tensor x = make_tensor_2d(
        2, 3,
        {1.0f, 2.0f, 3.0f,
         4.0f, 5.0f, 6.0f});

    Tensor y = layer.forward(x);

    EXPECT_FLOAT_EQ(y(0, 0), 32.0f);
    EXPECT_FLOAT_EQ(y(0, 1), 48.0f);
    EXPECT_FLOAT_EQ(y(1, 0), 59.0f);
    EXPECT_FLOAT_EQ(y(1, 1), 84.0f);
}

TEST_F(LinearForwardTest, BiasBroadcastsAcrossBatch) {
    Linear layer(/*in_features=*/2, /*out_features=*/3);

    layer.weight = make_tensor_2d(
        2, 3,
        {1.0f, 0.0f, 0.0f,
         0.0f, 1.0f, 0.0f},
        /*requires_grad=*/true);
    layer.bias = make_tensor_2d(1, 3, {10.0f, 20.0f, 30.0f}, /*requires_grad=*/true);

    Tensor x = make_tensor_2d(
        3, 2,
        {1.0f, 2.0f,
         3.0f, 4.0f,
         5.0f, 6.0f});

    Tensor y = layer.forward(x);

    EXPECT_FLOAT_EQ(y(0, 0), 11.0f);
    EXPECT_FLOAT_EQ(y(0, 1), 22.0f);
    EXPECT_FLOAT_EQ(y(0, 2), 30.0f);

    EXPECT_FLOAT_EQ(y(1, 0), 13.0f);
    EXPECT_FLOAT_EQ(y(1, 1), 24.0f);
    EXPECT_FLOAT_EQ(y(1, 2), 30.0f);

    EXPECT_FLOAT_EQ(y(2, 0), 15.0f);
    EXPECT_FLOAT_EQ(y(2, 1), 26.0f);
    EXPECT_FLOAT_EQ(y(2, 2), 30.0f);
}

TEST_F(LinearForwardTest, InputFeatureMismatchAsserts) {
    Linear layer(/*in_features=*/3, /*out_features=*/2);
    Tensor bad_x = Tensor::zeros(make_shape({4, 5}), 2);

    EXPECT_DEATH(layer.forward(bad_x), "");
}

class LinearAutogradTest : public ::testing::Test {};

TEST_F(LinearAutogradTest, BackwardPopulatesInputAndParameterGrads) {
    Linear layer(/*in_features=*/3, /*out_features=*/2);

    layer.weight = make_tensor_2d(
        3, 2,
        {1.0f, 2.0f,
         3.0f, 4.0f,
         5.0f, 6.0f},
        /*requires_grad=*/true);
    layer.bias = make_tensor_2d(1, 2, {10.0f, 20.0f}, /*requires_grad=*/true);

    Tensor x = make_tensor_2d(
        2, 3,
        {1.0f, 2.0f, 3.0f,
         4.0f, 5.0f, 6.0f},
        /*requires_grad=*/true);

    Tensor y = layer.forward(x);
    backward(y);

    ASSERT_TRUE(x.has_grad());
    ASSERT_TRUE(layer.weight.has_grad());
    ASSERT_TRUE(layer.bias.has_grad());

    EXPECT_FLOAT_EQ(x.grad()(0, 0), 3.0f);
    EXPECT_FLOAT_EQ(x.grad()(0, 1), 7.0f);
    EXPECT_FLOAT_EQ(x.grad()(0, 2), 11.0f);
    EXPECT_FLOAT_EQ(x.grad()(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(x.grad()(1, 1), 7.0f);
    EXPECT_FLOAT_EQ(x.grad()(1, 2), 11.0f);

    EXPECT_FLOAT_EQ(layer.weight.grad()(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(layer.weight.grad()(0, 1), 5.0f);
    EXPECT_FLOAT_EQ(layer.weight.grad()(1, 0), 7.0f);
    EXPECT_FLOAT_EQ(layer.weight.grad()(1, 1), 7.0f);
    EXPECT_FLOAT_EQ(layer.weight.grad()(2, 0), 9.0f);
    EXPECT_FLOAT_EQ(layer.weight.grad()(2, 1), 9.0f);

    EXPECT_FLOAT_EQ(layer.bias.grad()(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(layer.bias.grad()(0, 1), 2.0f);
}

TEST_F(LinearAutogradTest, InputWithoutGradDoesNotAccumulateGrad) {
    Linear layer(/*in_features=*/2, /*out_features=*/2);

    Tensor x = make_tensor_2d(
        3, 2,
        {1.0f, 2.0f,
         3.0f, 4.0f,
         5.0f, 6.0f},
        /*requires_grad=*/false);

    Tensor y = layer.forward(x);
    backward(y);

    EXPECT_FALSE(x.has_grad());
    EXPECT_TRUE(layer.weight.has_grad());
    EXPECT_TRUE(layer.bias.has_grad());
}
