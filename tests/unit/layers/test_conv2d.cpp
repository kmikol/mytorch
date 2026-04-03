#include <gtest/gtest.h>

#include <cstddef>
#include <initializer_list>
#include <vector>

#include "autograd.h"
#include "layers/conv2d.h"

namespace {

static Shape make_shape(std::initializer_list<size_t> dims) {
    Shape s{};
    size_t i = 0;
    for (size_t d : dims)
        s[i++] = d;
    return s;
}

static Tensor make_tensor(std::initializer_list<size_t> dims,
                          const std::vector<float>& vals,
                          bool requires_grad = false) {
    Shape s = make_shape(dims);
    const size_t ndim = dims.size();
    Tensor t(s, ndim, requires_grad);
    EXPECT_EQ(vals.size(), t.numel);

    for (size_t i = 0; i < vals.size(); ++i)
        t.storage->data[i] = vals[i];
    return t;
}

}  // namespace

class Conv2dCtorTest : public ::testing::Test {};

TEST_F(Conv2dCtorTest, ParameterShapesAndGradFlags) {
    Conv2d layer(
        /*in_channels=*/3,
        /*out_channels=*/4,
        /*kernel_h=*/3,
        /*kernel_w=*/5,
        /*stride_h=*/2,
        /*stride_w=*/1,
        /*padding_h=*/1,
        /*padding_w=*/2
    );

    EXPECT_EQ(layer.weight.ndim, 4u);
    EXPECT_EQ(layer.weight.shape_at(0), 4u);
    EXPECT_EQ(layer.weight.shape_at(1), 3u);
    EXPECT_EQ(layer.weight.shape_at(2), 3u);
    EXPECT_EQ(layer.weight.shape_at(3), 5u);

    EXPECT_EQ(layer.bias.ndim, 2u);
    EXPECT_EQ(layer.bias.shape_at(0), 1u);
    EXPECT_EQ(layer.bias.shape_at(1), 4u);

    EXPECT_TRUE(layer.weight.requires_grad());
    EXPECT_TRUE(layer.bias.requires_grad());
}

TEST_F(Conv2dCtorTest, ParametersReturnsWeightThenBias) {
    Conv2d layer(2, 3, 3, 3);
    auto params = layer.parameters();

    ASSERT_EQ(params.size(), 2u);
    EXPECT_EQ(params[0], &layer.weight);
    EXPECT_EQ(params[1], &layer.bias);
}

class Conv2dForwardTest : public ::testing::Test {};

TEST_F(Conv2dForwardTest, OutputShapeMatchesFormula) {
    Conv2d layer(
        /*in_channels=*/2,
        /*out_channels=*/5,
        /*kernel_h=*/3,
        /*kernel_w=*/3,
        /*stride_h=*/2,
        /*stride_w=*/2,
        /*padding_h=*/1,
        /*padding_w=*/1
    );

    Tensor x = Tensor::zeros(make_shape({4, 2, 7, 9}), 4);
    Tensor y = layer.forward(x);

    // out_h = (7 + 2*1 - 3)/2 + 1 = 4
    // out_w = (9 + 2*1 - 3)/2 + 1 = 5
    EXPECT_EQ(y.ndim, 4u);
    EXPECT_EQ(y.shape_at(0), 4u);
    EXPECT_EQ(y.shape_at(1), 5u);
    EXPECT_EQ(y.shape_at(2), 4u);
    EXPECT_EQ(y.shape_at(3), 5u);
}

TEST_F(Conv2dForwardTest, DeterministicValuesMatchReference) {
    Conv2d layer(
        /*in_channels=*/1,
        /*out_channels=*/1,
        /*kernel_h=*/2,
        /*kernel_w=*/2,
        /*stride_h=*/1,
        /*stride_w=*/1,
        /*padding_h=*/0,
        /*padding_w=*/0
    );

    layer.weight = make_tensor(
        {1, 1, 2, 2},
        {
            1.f, 0.f,
            0.f, 1.f
        },
        /*requires_grad=*/true
    );
    layer.bias = make_tensor({1, 1}, {0.5f}, /*requires_grad=*/true);

    Tensor x = make_tensor(
        {1, 1, 3, 3},
        {
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,
            7.f, 8.f, 9.f
        },
        /*requires_grad=*/false
    );

    Tensor y = layer.forward(x);

    // 2x2 kernel [[1,0],[0,1]] sums top-left and bottom-right of each patch + 0.5
    EXPECT_FLOAT_EQ(y(0, 0, 0, 0), 1.f + 5.f + 0.5f);
    EXPECT_FLOAT_EQ(y(0, 0, 0, 1), 2.f + 6.f + 0.5f);
    EXPECT_FLOAT_EQ(y(0, 0, 1, 0), 4.f + 8.f + 0.5f);
    EXPECT_FLOAT_EQ(y(0, 0, 1, 1), 5.f + 9.f + 0.5f);
}

class Conv2dAutogradTest : public ::testing::Test {};

TEST_F(Conv2dAutogradTest, BackwardPopulatesInputAndParameterGrads) {
    Conv2d layer(1, 2, 3, 3, 1, 1, 1, 1);
    Tensor x = Tensor::ones(make_shape({2, 1, 5, 5}), 4, /*requires_grad=*/true);

    Tensor y = layer.forward(x);
    backward(y);

    ASSERT_TRUE(x.has_grad());
    EXPECT_TRUE(layer.weight.has_grad());
    EXPECT_TRUE(layer.bias.has_grad());
}
