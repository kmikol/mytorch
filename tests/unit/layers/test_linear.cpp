#include <gtest/gtest.h>
#include <cstddef>
#include <cmath>

#include "layers/linear.h"

// -------------------------------------------------------------
// Helpers
// -------------------------------------------------------------

static Shape make_shape(std::initializer_list<size_t> dims) {
    Shape s{};
    size_t i = 0;
    for (size_t d : dims) s[i++] = d;
    return s;
}

static Tensor make_tensor(std::initializer_list<size_t> dims,
                          std::initializer_list<float> vals,
                          bool requires_grad = false) {
    Shape s = make_shape(dims);
    size_t ndim = dims.size();
    Tensor t(s, ndim, requires_grad);
    size_t i = 0;
    for (float v : vals) t.storage->data[i++] = v;
    return t;
}

// -------------------------------------------------------------
// Linear ctor / parameter definitions
// -------------------------------------------------------------

class LinearCtorTest : public ::testing::Test {};

TEST_F(LinearCtorTest, WeightAndBiasShapes) {
    Linear layer(/*in_features=*/3, /*out_features=*/2);

    EXPECT_EQ(layer.weight.ndim, 2u);
    EXPECT_EQ(layer.weight.shape_at(0), 3u);
    EXPECT_EQ(layer.weight.shape_at(1), 2u);

    EXPECT_EQ(layer.bias.ndim, 2u);
    EXPECT_EQ(layer.bias.shape_at(0), 1u);
    EXPECT_EQ(layer.bias.shape_at(1), 2u);
}

TEST_F(LinearCtorTest, ParametersRequireGrad) {
    Linear layer(/*in_features=*/4, /*out_features=*/5);
    EXPECT_TRUE(layer.weight.requires_grad());
    EXPECT_TRUE(layer.bias.requires_grad());
}

TEST_F(LinearCtorTest, BiasStartsAtZero) {
    Linear layer(/*in_features=*/5, /*out_features=*/4);
    for (size_t j = 0; j < layer.out_features; ++j)
        EXPECT_FLOAT_EQ(layer.bias(0, j), 0.f);
}

TEST_F(LinearCtorTest, XavierRangeForWeights) {
    size_t in_f = 6, out_f = 4;
    Linear layer(in_f, out_f);
    float limit = std::sqrt(6.f / static_cast<float>(in_f + out_f));

    for (size_t i = 0; i < layer.weight.numel; ++i) {
        float w = layer.weight.storage->data[i];
        EXPECT_LE(w, limit);
        EXPECT_GE(w, -limit);
    }
}

TEST_F(LinearCtorTest, ParametersReturnsWeightThenBias) {
    Linear layer(/*in_features=*/3, /*out_features=*/2);
    auto params = layer.parameters();

    ASSERT_EQ(params.size(), 2u);
    EXPECT_EQ(params[0], &layer.weight);
    EXPECT_EQ(params[1], &layer.bias);
}

// -------------------------------------------------------------
// Linear::forward
// -------------------------------------------------------------

class LinearForwardTest : public ::testing::Test {};

TEST_F(LinearForwardTest, OutputShapeIsBatchByOutFeatures) {
    Linear layer(/*in_features=*/3, /*out_features=*/2);
    auto x = make_tensor({4, 3}, {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,
        7.f, 8.f, 9.f,
        0.f, 1.f, 2.f
    });

    auto y = layer.forward(x);
    EXPECT_EQ(y.ndim, 2u);
    EXPECT_EQ(y.shape_at(0), 4u);
    EXPECT_EQ(y.shape_at(1), 2u);
}

TEST_F(LinearForwardTest, AffineValuesMatchReference) {
    // x: [2,3], W: [3,2], b: [1,2]
    // y = x @ W + b
    Linear layer(/*in_features=*/3, /*out_features=*/2);

    layer.weight = make_tensor({3, 2}, {
        1.f, 2.f,
        3.f, 4.f,
        5.f, 6.f
    }, /*requires_grad=*/true);
    layer.bias = make_tensor({1, 2}, {10.f, 20.f}, /*requires_grad=*/true);

    auto x = make_tensor({2, 3}, {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f
    });

    auto y = layer.forward(x);

    // Row 0: [1,2,3] @ W = [22,28], +b => [32,48]
    // Row 1: [4,5,6] @ W = [49,64], +b => [59,84]
    EXPECT_FLOAT_EQ(y(0, 0), 32.f);
    EXPECT_FLOAT_EQ(y(0, 1), 48.f);
    EXPECT_FLOAT_EQ(y(1, 0), 59.f);
    EXPECT_FLOAT_EQ(y(1, 1), 84.f);
}

TEST_F(LinearForwardTest, BiasBroadcastsAcrossBatch) {
    Linear layer(/*in_features=*/2, /*out_features=*/3);
    layer.weight = make_tensor({2, 3}, {
        1.f, 0.f, 0.f,
        0.f, 1.f, 0.f
    }, /*requires_grad=*/true);
    layer.bias = make_tensor({1, 3}, {10.f, 20.f, 30.f}, /*requires_grad=*/true);

    auto x = make_tensor({3, 2}, {
        1.f, 2.f,
        3.f, 4.f,
        5.f, 6.f
    });

    auto y = layer.forward(x);

    EXPECT_FLOAT_EQ(y(0, 0), 11.f);
    EXPECT_FLOAT_EQ(y(0, 1), 22.f);
    EXPECT_FLOAT_EQ(y(0, 2), 30.f);

    EXPECT_FLOAT_EQ(y(1, 0), 13.f);
    EXPECT_FLOAT_EQ(y(1, 1), 24.f);
    EXPECT_FLOAT_EQ(y(1, 2), 30.f);

    EXPECT_FLOAT_EQ(y(2, 0), 15.f);
    EXPECT_FLOAT_EQ(y(2, 1), 26.f);
    EXPECT_FLOAT_EQ(y(2, 2), 30.f);
}

TEST_F(LinearForwardTest, InputFeatureMismatchAsserts) {
    Linear layer(/*in_features=*/3, /*out_features=*/2);
    auto bad_x = Tensor::zeros(make_shape({4, 5}), 2);
    EXPECT_DEATH(layer.forward(bad_x), "");
}

// -------------------------------------------------------------
// End-to-end autograd through Linear::forward
// -------------------------------------------------------------

class LinearAutogradTest : public ::testing::Test {};

TEST_F(LinearAutogradTest, BackwardPopulatesInputAndParameterGrads) {
    Linear layer(/*in_features=*/3, /*out_features=*/2);
    layer.weight = make_tensor({3, 2}, {
        1.f, 2.f,
        3.f, 4.f,
        5.f, 6.f
    }, /*requires_grad=*/true);
    layer.bias = make_tensor({1, 2}, {10.f, 20.f}, /*requires_grad=*/true);

    auto x = make_tensor({2, 3}, {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f
    }, /*requires_grad=*/true);

    auto y = layer.forward(x);
    backward(y);

    ASSERT_TRUE(x.has_grad());
    ASSERT_TRUE(layer.weight.has_grad());
    ASSERT_TRUE(layer.bias.has_grad());

    // Seed is ones([2,2]).
    // dX = ones @ W^T, where W^T = [[1,3,5],[2,4,6]]
    // Each row: [3,7,11]
    EXPECT_FLOAT_EQ(x.grad()(0, 0), 3.f);
    EXPECT_FLOAT_EQ(x.grad()(0, 1), 7.f);
    EXPECT_FLOAT_EQ(x.grad()(0, 2), 11.f);
    EXPECT_FLOAT_EQ(x.grad()(1, 0), 3.f);
    EXPECT_FLOAT_EQ(x.grad()(1, 1), 7.f);
    EXPECT_FLOAT_EQ(x.grad()(1, 2), 11.f);

    // dW = X^T @ ones = [[5,5],[7,7],[9,9]]
    EXPECT_FLOAT_EQ(layer.weight.grad()(0, 0), 5.f);
    EXPECT_FLOAT_EQ(layer.weight.grad()(0, 1), 5.f);
    EXPECT_FLOAT_EQ(layer.weight.grad()(1, 0), 7.f);
    EXPECT_FLOAT_EQ(layer.weight.grad()(1, 1), 7.f);
    EXPECT_FLOAT_EQ(layer.weight.grad()(2, 0), 9.f);
    EXPECT_FLOAT_EQ(layer.weight.grad()(2, 1), 9.f);

    // dB = sum over batch of ones = [2,2]
    EXPECT_FLOAT_EQ(layer.bias.grad()(0, 0), 2.f);
    EXPECT_FLOAT_EQ(layer.bias.grad()(0, 1), 2.f);
}

TEST_F(LinearAutogradTest, InputWithoutGradDoesNotAccumulateGrad) {
    Linear layer(/*in_features=*/2, /*out_features=*/2);
    auto x = make_tensor({3, 2}, {
        1.f, 2.f,
        3.f, 4.f,
        5.f, 6.f
    }, /*requires_grad=*/false);

    auto y = layer.forward(x);
    backward(y);

    EXPECT_FALSE(x.has_grad());
    EXPECT_TRUE(layer.weight.has_grad());
    EXPECT_TRUE(layer.bias.has_grad());
}

TEST_F(LinearAutogradTest, ForwardValuesUnchangedAfterBackward) {
    Linear layer(/*in_features=*/2, /*out_features=*/2);
    layer.weight = make_tensor({2, 2}, {
        1.f, 2.f,
        3.f, 4.f
    }, /*requires_grad=*/true);
    layer.bias = make_tensor({1, 2}, {5.f, 6.f}, /*requires_grad=*/true);

    auto x = make_tensor({1, 2}, {7.f, 8.f}, /*requires_grad=*/true);

    auto y = layer.forward(x);
    float y00 = y(0, 0);
    float y01 = y(0, 1);

    backward(y);

    EXPECT_FLOAT_EQ(y(0, 0), y00);
    EXPECT_FLOAT_EQ(y(0, 1), y01);
}
