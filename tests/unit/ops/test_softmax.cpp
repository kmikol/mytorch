#include <gtest/gtest.h>
#include <cmath>
#include "tensorlib.h"
#include "ops/ops.h"

// ================================================================
// forward
// ================================================================

TEST(SoftmaxForward, OutputsArePositive) {
    Tensor x = Tensor::from_data({1.f, 2.f, 3.f}, {3, 1});
    Tensor s = softmax(x, 0);

    for (int64_t i = 0; i < 3; i++)
        EXPECT_GT(s.at(i, 0), 0.f) << "output at row " << i << " should be positive";
}

TEST(SoftmaxForward, OutputsSumToOne) {
    Tensor x = Tensor::from_data({1.f, 2.f, 3.f}, {3, 1});
    Tensor s = softmax(x, 0);

    float sum = 0.f;
    for (int64_t i = 0; i < 3; i++) sum += s.at(i, 0);
    EXPECT_NEAR(sum, 1.f, 1e-5f);
}

TEST(SoftmaxForward, OrderPreserved) {
    Tensor x = Tensor::from_data({1.f, 2.f, 3.f}, {3, 1});
    Tensor s = softmax(x, 0);

    EXPECT_GT(s.at(1, 0), s.at(0, 0));
    EXPECT_GT(s.at(2, 0), s.at(1, 0));
}

TEST(SoftmaxForward, KnownValues) {
    // softmax([0, 1]) = [1/(1+e), e/(1+e)] ≈ [0.2689, 0.7311]
    Tensor x = Tensor::from_data({0.f, 1.f}, {2, 1});
    Tensor s = softmax(x, 0);

    EXPECT_NEAR(s.at(0, 0), 0.2689f, 1e-3f);
    EXPECT_NEAR(s.at(1, 0), 0.7311f, 1e-3f);
}

TEST(SoftmaxForward, UniformInputGivesUniformOutput) {
    Tensor x = Tensor::from_data({2.f, 2.f, 2.f, 2.f}, {4, 1});
    Tensor s = softmax(x, 0);

    for (int64_t i = 0; i < 4; i++)
        EXPECT_NEAR(s.at(i, 0), 0.25f, 1e-5f);
}

TEST(SoftmaxForward, NumericalStabilityLargePositive) {
    Tensor x = Tensor::from_data({1000.f, 1001.f, 1002.f}, {3, 1});
    Tensor s = softmax(x, 0);

    float sum = 0.f;
    for (int64_t i = 0; i < 3; i++) {
        EXPECT_FALSE(std::isnan(s.at(i, 0)));
        sum += s.at(i, 0);
    }
    EXPECT_NEAR(sum, 1.f, 1e-5f);
}

TEST(SoftmaxForward, NumericalStabilityLargeNegative) {
    Tensor x = Tensor::from_data({-1000.f, -1001.f, -1002.f}, {3, 1});
    Tensor s = softmax(x, 0);

    float sum = 0.f;
    for (int64_t i = 0; i < 3; i++) {
        EXPECT_FALSE(std::isnan(s.at(i, 0)));
        sum += s.at(i, 0);
    }
    EXPECT_NEAR(sum, 1.f, 1e-5f);
}

TEST(SoftmaxForward, BatchedEachColumnIndependent) {
    Tensor x = Tensor::from_data(
        {1.f, 2.f, 3.f, 4.f,
         4.f, 3.f, 2.f, 1.f,
         2.f, 2.f, 2.f, 2.f},
        {3, 4});
    Tensor s = softmax(x, 0);

    for (int64_t n = 0; n < 4; n++) {
        float col_sum = 0.f;
        for (int64_t i = 0; i < 3; i++) col_sum += s.at(i, n);
        EXPECT_NEAR(col_sum, 1.f, 1e-5f) << "column " << n << " should sum to 1";
    }
}

TEST(SoftmaxForward, Dim1NormalisesAcrossColumns) {
    Tensor x = Tensor::from_data(
        {1.f, 2.f, 3.f,
         4.f, 5.f, 6.f},
        {2, 3});
    Tensor s = softmax(x, 1);

    for (int64_t r = 0; r < 2; r++) {
        float row_sum = 0.f;
        for (int64_t c = 0; c < 3; c++) row_sum += s.at(r, c);
        EXPECT_NEAR(row_sum, 1.f, 1e-5f) << "row " << r << " should sum to 1";
    }
}

// ================================================================
// backward
// ================================================================

// backward(s) seeds s.grad = ones(s.shape) and propagates back
// this is a valid upstream gradient — no loss function needed

TEST(SoftmaxBackward, GradientsSumToZero) {
    // softmax outputs sum to 1 — a fixed constraint
    // any perturbation must be zero-sum, so gradients always sum to 0
    // this holds for any upstream gradient, including all-ones
    Tensor x = Tensor::from_data({1.f, 2.f, 3.f}, {3, 1}, {}, true);
    Tensor s = softmax(x, 0);

    backward(s);   // seeds s.grad = ones([3,1])

    ASSERT_TRUE(x.has_grad());

    float sum = 0.f;
    for (int64_t i = 0; i < 3; i++) sum += x.grad().at(i, 0);
    EXPECT_NEAR(sum, 0.f, 1e-5f);
}

TEST(SoftmaxBackward, GradientShapeMatchesInput) {
    Tensor x = Tensor::from_data({1.f, 2.f, 3.f}, {3, 1}, {}, true);
    Tensor s = softmax(x, 0);

    backward(s);

    ASSERT_TRUE(x.has_grad());
    EXPECT_EQ(x.grad().shape(0), 3);
    EXPECT_EQ(x.grad().shape(1), 1);
}

TEST(SoftmaxBackward, KnownGradientValues) {
    // x = [1, 2], s ≈ [0.2689, 0.7311], upstream grad = [1, 1]
    //
    // dot = sum(grad * s) = 1*0.2689 + 1*0.7311 = 1.0
    //
    // dx_0 = s_0 * (grad_0 - dot) = 0.2689 * (1 - 1) = 0
    // dx_1 = s_1 * (grad_1 - dot) = 0.7311 * (1 - 1) = 0
    //
    // upstream all-ones gives zero gradient — makes sense:
    // scaling all softmax outputs equally has no effect on their
    // relative values, so the input gradient is zero
    Tensor x = Tensor::from_data({1.f, 2.f}, {2, 1}, {}, true);
    Tensor s = softmax(x, 0);

    backward(s);   // upstream = ones([2,1])

    ASSERT_TRUE(x.has_grad());
    EXPECT_NEAR(x.grad().at(0, 0), 0.f, 1e-5f);
    EXPECT_NEAR(x.grad().at(1, 0), 0.f, 1e-5f);
}

TEST(SoftmaxBackward, NonUniformUpstreamGrad) {
    // inject a non-uniform upstream gradient by manually setting
    // the softmax output's grad before calling its backward_fn directly
    //
    // x = [1, 2], s ≈ [0.2689, 0.7311], upstream grad = [1, 0]
    //
    // dot = 1*0.2689 + 0*0.7311 = 0.2689
    // dx_0 = 0.2689 * (1 - 0.2689) =  0.1966
    // dx_1 = 0.7311 * (0 - 0.2689) = -0.1966
    // sum(dx) = 0 ✓
    Tensor x = Tensor::from_data({1.f, 2.f}, {2, 1}, {}, true);
    Tensor s = softmax(x, 0);

    // set upstream gradient directly — no loss needed
    s.autograd_meta->grad = std::make_shared<Tensor>(
        Tensor::from_data({1.f, 0.f}, {2, 1}));

    // call the backward function manually — same thing backward() does
    // but we skip the seeding step since we already set the grad
    std::vector<Tensor> grads =
        s.autograd_meta->grad_fn->backward_fn(*s.autograd_meta->grad);

    // target class gets negative gradient (push logit up)
    EXPECT_NEAR(grads[0].at(0, 0),  0.1966f, 1e-3f);
    EXPECT_NEAR(grads[0].at(1, 0), -0.1966f, 1e-3f);

    // sum to zero
    float sum = grads[0].at(0, 0) + grads[0].at(1, 0);
    EXPECT_NEAR(sum, 0.f, 1e-5f);
}

TEST(SoftmaxBackward, BatchedGradientsSumToZeroPerColumn) {
    Tensor x = Tensor::from_data(
        {1.f, 2.f, 3.f, 4.f,
         4.f, 3.f, 2.f, 1.f,
         2.f, 2.f, 2.f, 2.f},
        {3, 4}, {}, true);

    Tensor s = softmax(x, 0);
    backward(s);   // upstream = ones([3,4])

    ASSERT_TRUE(x.has_grad());

    for (int64_t n = 0; n < 4; n++) {
        float col_sum = 0.f;
        for (int64_t i = 0; i < 3; i++)
            col_sum += x.grad().at(i, n);
        EXPECT_NEAR(col_sum, 0.f, 1e-5f)
            << "gradient sum for column " << n << " should be zero";
    }
}

TEST(SoftmaxBackward, NoGradWhenRequiresGradFalse) {
    Tensor x = Tensor::from_data({1.f, 2.f, 3.f}, {3, 1});
    Tensor s = softmax(x, 0);

    EXPECT_EQ(s.autograd_meta, nullptr);
}

TEST(SoftmaxBackward, GradFlowsThroughChain) {
    // sigmoid(softmax(x)) — gradient must flow all the way back to x
    Tensor x = Tensor::from_data({1.f, 2.f, 3.f}, {3, 1}, {}, true);
    Tensor s = softmax(x, 0);
    Tensor a = sigmoid(s);

    backward(a);   // seeds a.grad = ones([3,1])

    ASSERT_TRUE(x.has_grad());

    // sum-to-zero still holds after sigmoid because sigmoid is monotone
    // and doesn't break the zero-sum constraint from softmax
    float sum = 0.f;
    for (int64_t i = 0; i < 3; i++) sum += x.grad().at(i, 0);
    EXPECT_NEAR(sum, 0.f, 1e-4f);
}