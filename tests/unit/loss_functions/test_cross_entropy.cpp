#include <gtest/gtest.h>
#include <cstddef>
#include <cmath>

#include "loss_functions/cross_entropy.h"

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

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

// ─────────────────────────────────────────────
// CrossEntropyOp::forward — pure computation
// ─────────────────────────────────────────────

class CrossEntropyOpForwardTest : public ::testing::Test {};

TEST_F(CrossEntropyOpForwardTest, OutputIsScalar) {
    auto probs   = make_tensor({2, 3}, {0.1f, 0.7f, 0.2f, 0.3f, 0.3f, 0.4f});
    auto targets = make_tensor({2, 3}, {0.f,  1.f,  0.f,  1.f,  0.f,  0.f});
    auto loss    = CrossEntropyOp::forward(probs, targets);
    EXPECT_EQ(loss.ndim, 1u);
    EXPECT_EQ(loss.shape_at(0), 1u);
    EXPECT_EQ(loss.numel, 1u);
}

TEST_F(CrossEntropyOpForwardTest, PerfectPredictionIsNearZero) {
    // When predicted probability for the correct class is ~1, loss is ~0.
    auto probs   = make_tensor({1, 3}, {1.f - 2e-6f, 1e-6f, 1e-6f});
    auto targets = make_tensor({1, 3}, {1.f, 0.f, 0.f});
    auto loss    = CrossEntropyOp::forward(probs, targets);
    EXPECT_NEAR(loss(0), 0.f, 1e-5f);
}

TEST_F(CrossEntropyOpForwardTest, UniformPredictionOnOneHot) {
    // Uniform probs [1/3, 1/3, 1/3], one-hot target → loss = log(3) per sample.
    float p = 1.f / 3.f;
    auto probs   = make_tensor({1, 3}, {p, p, p});
    auto targets = make_tensor({1, 3}, {1.f, 0.f, 0.f});
    auto loss    = CrossEntropyOp::forward(probs, targets);
    EXPECT_NEAR(loss(0), std::log(3.f), 1e-5f);
}

TEST_F(CrossEntropyOpForwardTest, MeanOverBatch) {
    // Two identical samples — loss should equal the per-sample loss.
    float p = 1.f / 3.f;
    float ref = std::log(3.f);
    auto probs   = make_tensor({2, 3}, {p, p, p,  p, p, p});
    auto targets = make_tensor({2, 3}, {1.f, 0.f, 0.f,  1.f, 0.f, 0.f});
    auto loss    = CrossEntropyOp::forward(probs, targets);
    EXPECT_NEAR(loss(0), ref, 1e-5f);
}

TEST_F(CrossEntropyOpForwardTest, LossIsAlwaysNonNegative) {
    auto probs   = make_tensor({3, 2}, {0.8f, 0.2f,  0.5f, 0.5f,  0.1f, 0.9f});
    auto targets = make_tensor({3, 2}, {1.f,  0.f,   0.f,  1.f,   1.f,  0.f});
    auto loss    = CrossEntropyOp::forward(probs, targets);
    EXPECT_GE(loss(0), 0.f);
}

TEST_F(CrossEntropyOpForwardTest, ShapeMismatchAsserts) {
    auto probs   = Tensor::zeros(make_shape({2, 3}), 2);
    auto targets = Tensor::zeros(make_shape({2, 4}), 2);
    EXPECT_DEATH(CrossEntropyOp::forward(probs, targets), "");
}

TEST_F(CrossEntropyOpForwardTest, AlwaysProducesNoMeta) {
    auto probs   = make_tensor({1, 2}, {0.6f, 0.4f}, /*requires_grad=*/true);
    auto targets = make_tensor({1, 2}, {1.f, 0.f});
    auto loss    = CrossEntropyOp::forward(probs, targets);
    EXPECT_EQ(loss.autograd_meta, nullptr);
}

// ─────────────────────────────────────────────
// CrossEntropyOp::backward — pure gradient
// ─────────────────────────────────────────────

class CrossEntropyOpBackwardTest : public ::testing::Test {};

TEST_F(CrossEntropyOpBackwardTest, GradShape) {
    auto probs   = make_tensor({2, 3}, {0.1f, 0.7f, 0.2f, 0.3f, 0.3f, 0.4f});
    auto targets = make_tensor({2, 3}, {0.f,  1.f,  0.f,  1.f,  0.f,  0.f});
    auto grad    = make_tensor({1},   {1.f});
    auto grads   = CrossEntropyOp::backward(grad, probs, targets);

    EXPECT_EQ(grads[0].ndim, 2u);
    EXPECT_EQ(grads[0].shape_at(0), 2u);
    EXPECT_EQ(grads[0].shape_at(1), 3u);
}

TEST_F(CrossEntropyOpBackwardTest, ZeroTargetGivesZeroGrad) {
    // Where target is 0, gradient is 0 regardless of prob value.
    auto probs   = make_tensor({1, 3}, {0.2f, 0.5f, 0.3f});
    auto targets = make_tensor({1, 3}, {1.f,  0.f,  0.f});
    auto grad    = make_tensor({1},   {1.f});
    auto grads   = CrossEntropyOp::backward(grad, probs, targets);

    EXPECT_FLOAT_EQ(grads[0](0, 1), 0.f);
    EXPECT_FLOAT_EQ(grads[0](0, 2), 0.f);
}

TEST_F(CrossEntropyOpBackwardTest, GradNegativeForNonZeroTarget) {
    // dL/dprobs[i,j] = -targets[i,j] / (probs[i,j] * N) < 0 for targets > 0
    auto probs   = make_tensor({1, 3}, {0.2f, 0.5f, 0.3f});
    auto targets = make_tensor({1, 3}, {1.f,  0.f,  0.f});
    auto grad    = make_tensor({1},   {1.f});
    auto grads   = CrossEntropyOp::backward(grad, probs, targets);

    EXPECT_LT(grads[0](0, 0), 0.f);
}

TEST_F(CrossEntropyOpBackwardTest, GradScaledByUpstream) {
    // Doubling upstream gradient doubles all grad values.
    auto probs   = make_tensor({1, 2}, {0.6f, 0.4f});
    auto targets = make_tensor({1, 2}, {1.f,  0.f});
    auto grad1   = make_tensor({1},   {1.f});
    auto grad2   = make_tensor({1},   {2.f});

    auto g1 = CrossEntropyOp::backward(grad1, probs, targets);
    auto g2 = CrossEntropyOp::backward(grad2, probs, targets);

    EXPECT_NEAR(g2[0](0, 0), 2.f * g1[0](0, 0), 1e-6f);
}

TEST_F(CrossEntropyOpBackwardTest, NumericalValueCheck) {
    // N=1, targets=[1,0], probs=[p,1-p] → dL/dp_0 = -1/p, dL/dp_1 = 0
    float p = 0.7f;
    auto probs   = make_tensor({1, 2}, {p, 1.f - p});
    auto targets = make_tensor({1, 2}, {1.f, 0.f});
    auto grad    = make_tensor({1},   {1.f});
    auto grads   = CrossEntropyOp::backward(grad, probs, targets);

    float expected = -1.f / p;  // N=1, upstream=1
    EXPECT_NEAR(grads[0](0, 0), expected, 1e-5f);
    EXPECT_FLOAT_EQ(grads[0](0, 1), 0.f);
}

// ─────────────────────────────────────────────
// cross_entropy() — orchestration + autograd wiring
// ─────────────────────────────────────────────

class CrossEntropyFuncTest : public ::testing::Test {};

TEST_F(CrossEntropyFuncTest, ProducesCorrectValue) {
    float p = 1.f / 3.f;
    auto probs   = make_tensor({1, 3}, {p, p, p});
    auto targets = make_tensor({1, 3}, {1.f, 0.f, 0.f});
    auto loss    = cross_entropy(probs, targets);
    EXPECT_NEAR(loss(0), std::log(3.f), 1e-5f);
}

TEST_F(CrossEntropyFuncTest, NoRequiresGradProducesNoMeta) {
    auto probs   = make_tensor({1, 2}, {0.6f, 0.4f});
    auto targets = make_tensor({1, 2}, {1.f, 0.f});
    auto loss    = cross_entropy(probs, targets);
    EXPECT_EQ(loss.autograd_meta, nullptr);
}

TEST_F(CrossEntropyFuncTest, ProbsRequiresGradProducesMeta) {
    auto probs   = make_tensor({1, 2}, {0.6f, 0.4f}, /*requires_grad=*/true);
    auto targets = make_tensor({1, 2}, {1.f, 0.f});
    auto loss    = cross_entropy(probs, targets);
    EXPECT_TRUE(loss.requires_grad());
}

TEST_F(CrossEntropyFuncTest, TargetsRequiresGradGivesNoMeta) {
    // targets are data — cross_entropy never puts them in the graph.
    auto probs   = make_tensor({1, 2}, {0.6f, 0.4f});
    auto targets = make_tensor({1, 2}, {1.f, 0.f}, /*requires_grad=*/true);
    auto loss    = cross_entropy(probs, targets);
    EXPECT_EQ(loss.autograd_meta, nullptr);
}

// ─────────────────────────────────────────────
// End-to-end autograd via cross_entropy()
// ─────────────────────────────────────────────

class CrossEntropyAutogradTest : public ::testing::Test {};

TEST_F(CrossEntropyAutogradTest, GradFlowsToProbsOnly) {
    auto probs   = make_tensor({1, 3}, {0.2f, 0.5f, 0.3f}, /*requires_grad=*/true);
    auto targets = make_tensor({1, 3}, {1.f,  0.f,  0.f});

    auto loss = cross_entropy(probs, targets);
    backward(loss);

    ASSERT_TRUE(probs.has_grad());
}

TEST_F(CrossEntropyAutogradTest, GradNumericalValue) {
    // N=1, one-hot [1,0,0] → dL/dprobs[0,0] = -1/probs[0,0]
    float p0 = 0.6f;
    auto probs   = make_tensor({1, 3}, {p0, 0.3f, 0.1f}, /*requires_grad=*/true);
    auto targets = make_tensor({1, 3}, {1.f, 0.f,  0.f});

    auto loss = cross_entropy(probs, targets);
    backward(loss);

    EXPECT_NEAR(probs.grad()(0, 0), -1.f / p0, 1e-5f);
    EXPECT_FLOAT_EQ(probs.grad()(0, 1), 0.f);
    EXPECT_FLOAT_EQ(probs.grad()(0, 2), 0.f);
}

TEST_F(CrossEntropyAutogradTest, GradAveragedOverBatch) {
    // Same sample repeated N times — per-sample grad should be 1/N times the N=1 case.
    float p0 = 0.6f;
    size_t N = 4;
    std::vector<float> probs_data, tgt_data;
    for (size_t i = 0; i < N; ++i) { probs_data.push_back(p0); probs_data.push_back(0.4f); }
    for (size_t i = 0; i < N; ++i) { tgt_data.push_back(1.f);  tgt_data.push_back(0.f); }

    Tensor probs(make_shape({N, 2}), 2, /*requires_grad=*/true);
    Tensor targets(make_shape({N, 2}), 2);
    for (size_t row = 0; row < N; ++row) {
        probs(row, 0)   = probs_data[row * 2];
        probs(row, 1)   = probs_data[row * 2 + 1];
        targets(row, 0) = tgt_data[row * 2];
        targets(row, 1) = tgt_data[row * 2 + 1];
    }

    auto loss = cross_entropy(probs, targets);
    backward(loss);

    // dL/dprobs[i,0] = -1/(p0 * N) for all rows
    float expected = -1.f / (p0 * static_cast<float>(N));
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(probs.grad()(i, 0), expected, 1e-5f);
}
