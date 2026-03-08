#include <gtest/gtest.h>
#include <cmath>
#include "tensorlib.h"
#include "ops/ops.h"
#include "loss_functions/cross_entropy.h"

// ================================================================
// helpers
// ================================================================

// recompute softmax for a single column — used to verify known values
static std::vector<float> softmax_col(std::vector<float> v) {
    float max_val = *std::max_element(v.begin(), v.end());
    float sum = 0.f;
    for (float& x : v) { x = std::exp(x - max_val); sum += x; }
    for (float& x : v) x /= sum;
    return v;
}

// ================================================================
// forward
// ================================================================

TEST(CrossEntropyLossForward, OutputIsScalar) {
    // loss should always be a [1,1] tensor regardless of input size
    Tensor logits  = Tensor::from_data({1.f, 2.f, 3.f}, {3, 1});
    Tensor targets = Tensor::from_data({2.f}, {1, 1});

    Tensor loss = cross_entropy_loss(logits, targets);

    EXPECT_EQ(loss.shape(0), 1);
    EXPECT_EQ(loss.shape(1), 1);
}

TEST(CrossEntropyLossForward, LossIsNonNegative) {
    // cross-entropy is always >= 0
    Tensor logits  = Tensor::from_data({2.f, 1.f, 0.5f}, {3, 1});
    Tensor targets = Tensor::from_data({0.f}, {1, 1});

    Tensor loss = cross_entropy_loss(logits, targets);

    EXPECT_GE(loss.at({0, 0}), 0.f);
}

TEST(CrossEntropyLossForward, KnownValueSingleSample) {
    // manual calculation:
    // logits = [1, 0], target = 0
    // max = 1
    // sum_exp = exp(0) + exp(-1) = 1 + 0.3679 = 1.3679
    // log_normaliser = log(1.3679) + 1 = 0.3133 + 1 = 1.3133
    // loss = -logits[0] + log_normaliser = -1 + 1.3133 = 0.3133
    Tensor logits  = Tensor::from_data({1.f, 0.f}, {2, 1});
    Tensor targets = Tensor::from_data({0.f}, {1, 1});

    Tensor loss = cross_entropy_loss(logits, targets);

    EXPECT_NEAR(loss.at({0, 0}), 0.3133f, 1e-3f);
}

TEST(CrossEntropyLossForward, UniformLogitsGiveLogC) {
    // when all logits are equal, softmax is uniform (1/C each)
    // cross-entropy = -log(1/C) = log(C)
    // for C=4: log(4) ≈ 1.3863
    Tensor logits  = Tensor::from_data({0.f, 0.f, 0.f, 0.f}, {4, 1});
    Tensor targets = Tensor::from_data({2.f}, {1, 1});

    Tensor loss = cross_entropy_loss(logits, targets);

    EXPECT_NEAR(loss.at({0, 0}), std::log(4.f), 1e-5f);
}

TEST(CrossEntropyLossForward, PerfectPredictionGivesLowLoss) {
    // when the correct class has a very large logit, loss approaches 0
    Tensor logits  = Tensor::from_data({100.f, -100.f, -100.f}, {3, 1});
    Tensor targets = Tensor::from_data({0.f}, {1, 1});

    Tensor loss = cross_entropy_loss(logits, targets);

    EXPECT_NEAR(loss.at({0, 0}), 0.f, 1e-3f);
}

TEST(CrossEntropyLossForward, WrongPredictionGivesHighLoss) {
    // when the correct class has a very low logit, loss is large
    Tensor logits  = Tensor::from_data({-100.f, 100.f, -100.f}, {3, 1});
    Tensor targets = Tensor::from_data({0.f}, {1, 1});

    Tensor loss = cross_entropy_loss(logits, targets);

    EXPECT_GT(loss.at({0, 0}), 10.f);
}

TEST(CrossEntropyLossForward, BatchedLossIsMean) {
    // batched loss = mean of per-sample losses
    // compute each individually and verify mean matches batched
    // sample 0: logits=[2,1], target=0
    // sample 1: logits=[1,2], target=1
    Tensor logits_0 = Tensor::from_data({2.f, 1.f}, {2, 1});
    Tensor logits_1 = Tensor::from_data({1.f, 2.f}, {2, 1});
    Tensor targets_0 = Tensor::from_data({0.f}, {1, 1});
    Tensor targets_1 = Tensor::from_data({1.f}, {1, 1});

    float loss_0 = cross_entropy_loss(logits_0, targets_0).at({0, 0});
    float loss_1 = cross_entropy_loss(logits_1, targets_1).at({0, 0});
    float expected_mean = (loss_0 + loss_1) / 2.f;

    // batched version
    Tensor logits_batch  = Tensor::from_data({2.f, 1.f,
                                              1.f, 2.f}, {2, 2});
    Tensor targets_batch = Tensor::from_data({0.f, 1.f}, {1, 2});

    float batched_loss = cross_entropy_loss(logits_batch, targets_batch).at({0, 0});

    EXPECT_NEAR(batched_loss, expected_mean, 1e-5f);
}

TEST(CrossEntropyLossForward, NumericalStabilityLargeLogits) {
    // without max subtraction, exp(1000) overflows to inf
    Tensor logits  = Tensor::from_data({1000.f, 1001.f, 1002.f}, {3, 1});
    Tensor targets = Tensor::from_data({2.f}, {1, 1});

    Tensor loss = cross_entropy_loss(logits, targets);

    EXPECT_FALSE(std::isnan(loss.at({0, 0})));
    EXPECT_FALSE(std::isinf(loss.at({0, 0})));
    EXPECT_GE(loss.at({0, 0}), 0.f);
}

TEST(CrossEntropyLossForward, NumericalStabilityNegativeLogits) {
    // without max subtraction, exp(-1000) underflows to 0
    // causing log(0) = -inf
    Tensor logits  = Tensor::from_data({-1000.f, -1001.f, -1002.f}, {3, 1});
    Tensor targets = Tensor::from_data({0.f}, {1, 1});

    Tensor loss = cross_entropy_loss(logits, targets);

    EXPECT_FALSE(std::isnan(loss.at({0, 0})));
    EXPECT_FALSE(std::isinf(loss.at({0, 0})));
}

// ================================================================
// backward
// ================================================================

TEST(CrossEntropyLossBackward, GradientShape) {
    // gradient must have same shape as logits
    Tensor logits  = Tensor::from_data({1.f, 2.f, 0.f}, {3, 1}, {}, true);
    Tensor targets = Tensor::from_data({1.f}, {1, 1});

    Tensor loss = cross_entropy_loss(logits, targets);
    backward(loss);

    ASSERT_TRUE(logits.has_grad());
    EXPECT_EQ(logits.grad().shape(0), 3);
    EXPECT_EQ(logits.grad().shape(1), 1);
}

TEST(CrossEntropyLossBackward, GradientSumsToZero) {
    // key property: gradients sum to zero over the class dimension
    // because softmax outputs sum to 1, and d/d_logits of a constant = 0
    // (softmax - one_hot) sums to (1 - 1) = 0 per sample
    Tensor logits  = Tensor::from_data({1.f, 2.f, 3.f}, {3, 1}, {}, true);
    Tensor targets = Tensor::from_data({0.f}, {1, 1});

    Tensor loss = cross_entropy_loss(logits, targets);
    backward(loss);

    ASSERT_TRUE(logits.has_grad());

    float sum = 0.f;
    for (int64_t i = 0; i < 3; i++)
        sum += logits.grad().at({i, 0});

    EXPECT_NEAR(sum, 0.f, 1e-5f);
}

TEST(CrossEntropyLossBackward, BatchedGradientSumsToZeroPerSample) {
    // gradient sum-to-zero must hold independently per sample (column)
    Tensor logits  = Tensor::from_data({1.f, 2.f, 2.f, 1.f,
                                        3.f, 1.f, 1.f, 3.f,
                                        2.f, 3.f, 3.f, 2.f}, {3, 4}, {}, true);
    Tensor targets = Tensor::from_data({0.f, 1.f, 2.f, 0.f}, {1, 4});

    Tensor loss = cross_entropy_loss(logits, targets);
    backward(loss);

    ASSERT_TRUE(logits.has_grad());

    for (int64_t n = 0; n < 4; n++) {
        float col_sum = 0.f;
        for (int64_t c = 0; c < 3; c++)
            col_sum += logits.grad().at({c, n});
        EXPECT_NEAR(col_sum, 0.f, 1e-5f)
            << "gradient sum for sample " << n << " should be zero";
    }
}

TEST(CrossEntropyLossBackward, KnownGradientValues) {
    // manual calculation:
    // logits = [1, 0], target = 0
    // softmax ≈ [0.7311, 0.2689]
    // gradient = (softmax - one_hot) / N
    // = ([0.7311, 0.2689] - [1, 0]) / 1
    // = [-0.2689,  0.2689]
    Tensor logits  = Tensor::from_data({1.f, 0.f}, {2, 1}, {}, true);
    Tensor targets = Tensor::from_data({0.f}, {1, 1});

    Tensor loss = cross_entropy_loss(logits, targets);
    backward(loss);

    ASSERT_TRUE(logits.has_grad());

    // target class gradient is negative (push logit higher)
    EXPECT_LT(logits.grad().at({0, 0}), 0.f);

    // non-target class gradient is positive (push logit lower)
    EXPECT_GT(logits.grad().at({1, 0}), 0.f);

    // known values
    EXPECT_NEAR(logits.grad().at({0, 0}), -0.2689f, 1e-3f);
    EXPECT_NEAR(logits.grad().at({1, 0}),  0.2689f, 1e-3f);
}

TEST(CrossEntropyLossBackward, PerfectPredictionSmallGradient) {
    // when logits strongly favour the correct class,
    // softmax ≈ one_hot, so gradient ≈ 0
    Tensor logits  = Tensor::from_data({100.f, -100.f}, {2, 1}, {}, true);
    Tensor targets = Tensor::from_data({0.f}, {1, 1});

    Tensor loss = cross_entropy_loss(logits, targets);
    backward(loss);

    ASSERT_TRUE(logits.has_grad());

    EXPECT_NEAR(logits.grad().at({0, 0}), 0.f, 1e-3f);
    EXPECT_NEAR(logits.grad().at({1, 0}), 0.f, 1e-3f);
}

TEST(CrossEntropyLossBackward, NoGradWhenRequiresGradFalse) {
    // no autograd_meta attached when input doesn't require grad
    Tensor logits  = Tensor::from_data({1.f, 2.f, 3.f}, {3, 1});
    Tensor targets = Tensor::from_data({1.f}, {1, 1});

    Tensor loss = cross_entropy_loss(logits, targets);

    EXPECT_EQ(loss.autograd_meta, nullptr);
}

TEST(CrossEntropyLossBackward, GradFlowsThroughLinearLayer) {
    // verify grad flows through a linear layer into its weights
    // logits = W @ x, loss = cross_entropy(logits, target)
    // W should receive a gradient
    Tensor W = Tensor::from_data(
        {0.1f, 0.2f, 0.3f, 0.4f,
         0.5f, 0.6f, 0.7f, 0.8f,
         0.9f, 1.0f, 1.1f, 1.2f},
        {3, 4}, {}, true);

    Tensor x       = Tensor::from_data({1.f, 0.f, 1.f, 0.f}, {4, 1});
    Tensor targets = Tensor::from_data({2.f}, {1, 1});

    Tensor logits = matmul(W, x);
    Tensor loss   = cross_entropy_loss(logits, targets);
    backward(loss);

    ASSERT_TRUE(W.has_grad());
    EXPECT_EQ(W.grad().shape(0), 3);
    EXPECT_EQ(W.grad().shape(1), 4);

    // W grad should also sum to zero per column (from the sum-to-zero property)
    // W.grad = d_logits @ x^T, and sum of d_logits = 0
    // so each column of W.grad sums to 0
    for (int64_t col = 0; col < 4; col++) {
        float col_sum = 0.f;
        for (int64_t row = 0; row < 3; row++)
            col_sum += W.grad().at({row, col});
        EXPECT_NEAR(col_sum, 0.f, 1e-5f)
            << "W gradient column " << col << " should sum to zero";
    }
}

TEST(CrossEntropyLossBackward, LossDecreasesAfterGradientStep) {
    // applying one gradient step should reduce the loss
    float lr = 0.1f;

    Tensor logits  = Tensor::from_data({1.f, 2.f, 3.f}, {3, 1}, {}, true);
    Tensor targets = Tensor::from_data({0.f}, {1, 1});

    Tensor loss_before = cross_entropy_loss(logits, targets);
    float  val_before  = loss_before.at({0, 0});

    backward(loss_before);
    ASSERT_TRUE(logits.has_grad());

    // manual SGD step
    for (int64_t i = 0; i < 3; i++)
        logits.at({i, 0}) -= lr * logits.grad().at({i, 0});

    // clear grad and recompute
    logits.autograd_meta->grad = nullptr;

    Tensor loss_after = cross_entropy_loss(logits, targets);
    float  val_after  = loss_after.at({0, 0});

    EXPECT_LT(val_after, val_before)
        << "loss should decrease after one gradient step";
}
