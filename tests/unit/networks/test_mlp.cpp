#include <gtest/gtest.h>

#include <cstddef>
#include <initializer_list>
#include <vector>

#include "autograd.h"
#include "networks/mlp.h"
#include "ops/activations/relu.h"

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
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j)
            t(i, j) = values[idx++];
    }
    return t;
}

}  // namespace

class MLPCtorTest : public ::testing::Test {};

TEST_F(MLPCtorTest, ParametersContainAllLayerWeightsAndBiases) {
    MLP model(/*input_features=*/3, /*hidden_layer_features=*/{5, 4}, relu, /*output_features=*/2);
    auto params = model.parameters();

    // 3 linear layers -> (weight,bias) x 3 = 6 tensors.
    ASSERT_EQ(params.size(), 6u);

    EXPECT_EQ(params[0]->shape_at(0), 3u);
    EXPECT_EQ(params[0]->shape_at(1), 5u);
    EXPECT_EQ(params[1]->shape_at(0), 1u);
    EXPECT_EQ(params[1]->shape_at(1), 5u);

    EXPECT_EQ(params[2]->shape_at(0), 5u);
    EXPECT_EQ(params[2]->shape_at(1), 4u);
    EXPECT_EQ(params[3]->shape_at(0), 1u);
    EXPECT_EQ(params[3]->shape_at(1), 4u);

    EXPECT_EQ(params[4]->shape_at(0), 4u);
    EXPECT_EQ(params[4]->shape_at(1), 2u);
    EXPECT_EQ(params[5]->shape_at(0), 1u);
    EXPECT_EQ(params[5]->shape_at(1), 2u);
}

class MLPForwardTest : public ::testing::Test {};

TEST_F(MLPForwardTest, OutputShapeIsBatchByOutputFeatures) {
    MLP model(/*input_features=*/3, /*hidden_layer_features=*/{4, 2}, relu, /*output_features=*/5);
    Tensor x = make_tensor_2d(
        7, 3,
        {1.0f, 2.0f, 3.0f,
         4.0f, 5.0f, 6.0f,
         7.0f, 8.0f, 9.0f,
         1.0f, 0.0f, 1.0f,
         2.0f, 2.0f, 2.0f,
         0.0f, 1.0f, 0.0f,
         3.0f, 1.0f, 4.0f});

    Tensor y = model.forward(x);
    EXPECT_EQ(y.ndim, 2u);
    EXPECT_EQ(y.shape_at(0), 7u);
    EXPECT_EQ(y.shape_at(1), 5u);
}

TEST_F(MLPForwardTest, DeterministicTwoLayerForwardMatchesReference) {
    MLP model(/*input_features=*/2, /*hidden_layer_features=*/{2}, relu, /*output_features=*/1);
    auto params = model.parameters();
    ASSERT_EQ(params.size(), 4u);

    *params[0] = make_tensor_2d(
        2, 2,
        {1.0f, 0.0f,
         0.0f, 1.0f},
        /*requires_grad=*/true);
    *params[1] = make_tensor_2d(1, 2, {0.0f, 1.0f}, /*requires_grad=*/true);

    *params[2] = make_tensor_2d(2, 1, {2.0f, 3.0f}, /*requires_grad=*/true);
    *params[3] = make_tensor_2d(1, 1, {0.5f}, /*requires_grad=*/true);

    Tensor x = make_tensor_2d(
        2, 2,
        {1.0f, 2.0f,
         -1.0f, 4.0f},
        /*requires_grad=*/false);

    // hidden = relu(x @ I + [0,1])
    // row0: relu([1,3]) = [1,3]
    // row1: relu([-1,5]) = [0,5]
    // out = hidden @ [2,3] + 0.5
    // row0: 1*2 + 3*3 + 0.5 = 11.5
    // row1: 0*2 + 5*3 + 0.5 = 15.5
    Tensor y = model.forward(x);

    EXPECT_EQ(y.shape_at(0), 2u);
    EXPECT_EQ(y.shape_at(1), 1u);
    EXPECT_FLOAT_EQ(y(0, 0), 11.5f);
    EXPECT_FLOAT_EQ(y(1, 0), 15.5f);
}

class MLPAutogradTest : public ::testing::Test {};

TEST_F(MLPAutogradTest, BackwardPopulatesInputAndAllParameterGrads) {
    MLP model(/*input_features=*/2, /*hidden_layer_features=*/{3}, relu, /*output_features=*/2);
    Tensor x = make_tensor_2d(
        3, 2,
        {1.0f, 2.0f,
         3.0f, 4.0f,
         5.0f, 6.0f},
        /*requires_grad=*/true);

    Tensor y = model.forward(x);
    backward(y);

    ASSERT_TRUE(x.has_grad());

    auto params = model.parameters();
    ASSERT_EQ(params.size(), 4u);
    for (Tensor* p : params)
        EXPECT_TRUE(p->has_grad());
}
