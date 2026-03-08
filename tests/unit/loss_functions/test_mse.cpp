#include <gtest/gtest.h>
#include "tensorlib.h"
#include "ops/ops.h"
#include "loss_functions/mse.h"

// ════════════════════════════════════════════════════════════════════════════
// MSE Loss Unit Tests
// ════════════════════════════════════════════════════════════════════════════

TEST(MSELossTest, ScalarInputForward) {
    Tensor pred   = Tensor::from_data({2.0f}, {1,1});
    Tensor target = Tensor::from_data({3.0f}, {1,1});

    Tensor loss = mse_loss(pred, target);

    EXPECT_EQ(loss.shape(0), 1);
    EXPECT_EQ(loss.shape(1), 1);
    EXPECT_FLOAT_EQ(loss.at({0,0}), 1.0f); // (2-3)^2 = 1
}

TEST(MSELossTest, ScalarInputBackward) {
    Tensor pred   = Tensor::from_data({2.0f}, {1,1}, {}, true);
    Tensor target = Tensor::from_data({3.0f}, {1,1}, {}, false);

    Tensor loss = mse_loss(pred, target);
    backward(loss);

    ASSERT_TRUE(pred.has_grad());
    EXPECT_FLOAT_EQ(pred.grad().at({0,0}), 2.0f * (2.0f - 3.0f) / 1.0f); // 2*(2-3)/n=2*(2-3)/1=-2
}

TEST(MSELossTest, VectorInputForward) {
    Tensor pred   = Tensor::from_data({1.0f, 2.0f, 3.0f}, {1,3});
    Tensor target = Tensor::from_data({2.0f, 2.0f, 2.0f}, {1,3});

    Tensor loss = mse_loss(pred, target);

    float expected = ((1-2)*(1-2) + (2-2)*(2-2) + (3-2)*(3-2)) / 3.0f;
    EXPECT_FLOAT_EQ(loss.at({0,0}), expected);
}

TEST(MSELossTest, VectorInputBackward) {
    Tensor pred   = Tensor::from_data({1.0f, 2.0f, 3.0f}, {1,3}, {}, true);
    Tensor target = Tensor::from_data({2.0f, 2.0f, 2.0f}, {1,3}, {}, false);

    Tensor loss = mse_loss(pred, target);
    backward(loss);

    ASSERT_TRUE(pred.has_grad());
    EXPECT_FLOAT_EQ(pred.grad().at({0,0}), 2.0f*(1-2)/3.0f);
    EXPECT_FLOAT_EQ(pred.grad().at({0,1}), 2.0f*(2-2)/3.0f);
    EXPECT_FLOAT_EQ(pred.grad().at({0,2}), 2.0f*(3-2)/3.0f);
}

TEST(MSELossTest, Batched2DForward) {
    Tensor pred   = Tensor::from_data({1,2,3,4,5,6}, {2,3});
    Tensor target = Tensor::from_data({1,1,1,1,1,1}, {2,3});

    Tensor loss = mse_loss(pred, target);

    float sum_sq = (1-1)*(1-1) + (2-1)*(2-1) + (3-1)*(3-1)
                 + (4-1)*(4-1) + (5-1)*(5-1) + (6-1)*(6-1);
    float expected = sum_sq / 6.0f;

    EXPECT_FLOAT_EQ(loss.at({0,0}), expected);
}

TEST(MSELossTest, Batched2DBackward) {
    Tensor pred   = Tensor::from_data({1,2,3,4,5,6}, {2,3}, {}, true);
    Tensor target = Tensor::from_data({1,1,1,1,1,1}, {2,3}, {}, false);

    Tensor loss = mse_loss(pred, target);
    backward(loss);

    ASSERT_TRUE(pred.has_grad());

    float n = 6.0f;
    EXPECT_FLOAT_EQ(pred.grad().at({0,0}), 2*(1-1)/n);
    EXPECT_FLOAT_EQ(pred.grad().at({0,1}), 2*(2-1)/n);
    EXPECT_FLOAT_EQ(pred.grad().at({0,2}), 2*(3-1)/n);
    EXPECT_FLOAT_EQ(pred.grad().at({1,0}), 2*(4-1)/n);
    EXPECT_FLOAT_EQ(pred.grad().at({1,1}), 2*(5-1)/n);
    EXPECT_FLOAT_EQ(pred.grad().at({1,2}), 2*(6-1)/n);
}
