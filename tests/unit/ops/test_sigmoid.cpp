#include <gtest/gtest.h>
#include <cstddef>
#include <cmath>

#include "ops/activations/sigmoid.h"

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

static float ref_sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

// ─────────────────────────────────────────────
// SigmoidOp::forward — pure computation
// ─────────────────────────────────────────────

class SigmoidOpForwardTest : public ::testing::Test {};

TEST_F(SigmoidOpForwardTest, ZeroGivesHalf) {
    auto x   = make_tensor({1}, {0.f});
    auto out = SigmoidOp::forward(x);
    EXPECT_FLOAT_EQ(out(0), 0.5f);
}

TEST_F(SigmoidOpForwardTest, LargePositiveApproachesOne) {
    auto x   = make_tensor({1}, {100.f});
    auto out = SigmoidOp::forward(x);
    EXPECT_NEAR(out(0), 1.f, 1e-5f);
}

TEST_F(SigmoidOpForwardTest, LargeNegativeApproachesZero) {
    auto x   = make_tensor({1}, {-100.f});
    auto out = SigmoidOp::forward(x);
    EXPECT_NEAR(out(0), 0.f, 1e-5f);
}

TEST_F(SigmoidOpForwardTest, ValuesMatchFormula) {
    auto x   = make_tensor({5}, {-2.f, -1.f, 0.f, 1.f, 2.f});
    auto out = SigmoidOp::forward(x);
    for (size_t i = 0; i < 5; ++i)
        EXPECT_NEAR(out(i), ref_sigmoid(x(i)), 1e-6f);
}

TEST_F(SigmoidOpForwardTest, OutputInOpenUnitInterval) {
    auto x   = make_tensor({4}, {-10.f, -1.f, 1.f, 10.f});
    auto out = SigmoidOp::forward(x);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_GT(out(i), 0.f);
        EXPECT_LT(out(i), 1.f);
    }
}

TEST_F(SigmoidOpForwardTest, OutputShapePreserved) {
    auto out = SigmoidOp::forward(Tensor::zeros(make_shape({3, 4}), 2));
    EXPECT_EQ(out.ndim, 2u);
    EXPECT_EQ(out.shape_at(0), 3u);
    EXPECT_EQ(out.shape_at(1), 4u);
}

TEST_F(SigmoidOpForwardTest, AlwaysProducesNoMeta) {
    auto x   = make_tensor({4}, {1.f, 2.f, 3.f, 4.f}, /*requires_grad=*/true);
    auto out = SigmoidOp::forward(x);
    EXPECT_EQ(out.autograd_meta, nullptr);
}

// ─────────────────────────────────────────────
// SigmoidOp::backward — pure gradient computation
// ─────────────────────────────────────────────

class SigmoidOpBackwardTest : public ::testing::Test {};

TEST_F(SigmoidOpBackwardTest, UnitGradGivesDerivative) {
    // With unit upstream: dL/dx = out * (1 - out)
    auto x    = make_tensor({4}, {-2.f, -1.f, 0.f, 2.f});
    auto out  = SigmoidOp::forward(x);
    auto grad = Tensor::ones(make_shape({4}), 1);
    auto gx   = SigmoidOp::backward(grad, out);

    for (size_t i = 0; i < 4; ++i) {
        float o   = out(i);
        float ref = o * (1.f - o);
        EXPECT_NEAR(gx(i), ref, 1e-6f);
    }
}

TEST_F(SigmoidOpBackwardTest, MaxDerivativeAtZero) {
    // σ'(0) = 0.5 * 0.5 = 0.25 — maximum of the sigmoid derivative
    auto out  = make_tensor({1}, {0.5f});  // σ(0) = 0.5
    auto grad = Tensor::ones(make_shape({1}), 1);
    auto gx   = SigmoidOp::backward(grad, out);
    EXPECT_NEAR(gx(0), 0.25f, 1e-6f);
}

TEST_F(SigmoidOpBackwardTest, DerivativeNearZeroForExtreme) {
    // σ(x) ≈ 0 or 1 at extremes → σ'(x) ≈ 0
    auto x    = make_tensor({2}, {-10.f, 10.f});
    auto out  = SigmoidOp::forward(x);
    auto grad = Tensor::ones(make_shape({2}), 1);
    auto gx   = SigmoidOp::backward(grad, out);
    EXPECT_NEAR(gx(0), 0.f, 1e-4f);
    EXPECT_NEAR(gx(1), 0.f, 1e-4f);
}

TEST_F(SigmoidOpBackwardTest, UpstreamGradScales) {
    auto out  = make_tensor({1}, {0.5f});  // σ(0)
    auto grad = make_tensor({1}, {4.f});
    auto gx   = SigmoidOp::backward(grad, out);
    // dL/dx = 4 * 0.5 * 0.5 = 1.0
    EXPECT_NEAR(gx(0), 1.f, 1e-6f);
}

// ─────────────────────────────────────────────
// sigmoid() — orchestration + autograd wiring
// ─────────────────────────────────────────────

class SigmoidFuncTest : public ::testing::Test {};

TEST_F(SigmoidFuncTest, ProducesCorrectValues) {
    auto x   = make_tensor({3}, {-1.f, 0.f, 1.f});
    auto out = sigmoid(x);
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(out(i), ref_sigmoid(x(i)), 1e-6f);
}

TEST_F(SigmoidFuncTest, NoRequiresGradProducesNoMeta) {
    auto out = sigmoid(make_tensor({4}, {1.f, 2.f, 3.f, 4.f}));
    EXPECT_EQ(out.autograd_meta, nullptr);
}

TEST_F(SigmoidFuncTest, RequiresGradProducesMeta) {
    auto out = sigmoid(make_tensor({4}, {1.f, 2.f, 3.f, 4.f}, /*requires_grad=*/true));
    EXPECT_TRUE(out.requires_grad());
}

// ─────────────────────────────────────────────
// End-to-end autograd via sigmoid()
// ─────────────────────────────────────────────

class SigmoidAutogradTest : public ::testing::Test {};

TEST_F(SigmoidAutogradTest, GradMatchesDerivative) {
    // z = sigmoid(x), seed=ones → dL/dx = σ(x)(1 - σ(x))
    auto x = make_tensor({4}, {-2.f, -1.f, 0.f, 2.f}, /*requires_grad=*/true);
    auto z = sigmoid(x);
    backward(z);

    ASSERT_TRUE(x.has_grad());
    for (size_t i = 0; i < 4; ++i) {
        float xi  = x(i);
        float sig = ref_sigmoid(xi);
        EXPECT_NEAR(x.grad()(i), sig * (1.f - sig), 1e-6f);
    }
}

TEST_F(SigmoidAutogradTest, ForwardValuesUnchangedByBackward) {
    auto x = make_tensor({3}, {-1.f, 0.f, 1.f}, /*requires_grad=*/true);
    auto z = sigmoid(x);
    float v0 = z(0);
    float v2 = z(2);
    backward(z);
    EXPECT_FLOAT_EQ(z(0), v0);
    EXPECT_FLOAT_EQ(z(2), v2);
}

// ─────────────────────────────────────────────
// Non-contiguous inputs (stride-aware paths)
// ─────────────────────────────────────────────

class SigmoidNonContiguousTest : public ::testing::Test {};

// A transposed view exercises the stride-aware slow path in SigmoidOp::forward.
TEST_F(SigmoidNonContiguousTest, ForwardOnTransposedInput) {
    // x = [[0, 100],[-100, 0]]  shape [2,2]
    // x.T() = [[0,-100],[100,0]]  shape [2,2], non-contiguous
    // sigmoid(x.T()): 0→0.5, -100≈0, 100≈1, 0→0.5
    auto x = Tensor::zeros(make_shape({2, 2}), 2);
    x(0,0)= 0.f; x(0,1)= 100.f;
    x(1,0)=-100.f; x(1,1)= 0.f;
    auto xt = x.T();

    ASSERT_FALSE(xt.is_contiguous());
    auto out = SigmoidOp::forward(xt);

    EXPECT_NEAR(out(0,0), 0.5f,  1e-5f);   // sigmoid(0)
    EXPECT_NEAR(out(0,1), 0.f,   1e-4f);   // sigmoid(-100) ≈ 0
    EXPECT_NEAR(out(1,0), 1.f,   1e-4f);   // sigmoid(100)  ≈ 1
    EXPECT_NEAR(out(1,1), 0.5f,  1e-5f);   // sigmoid(0)
}

// ─────────────────────────────────────────────
// Boundary / numerical stability
// ─────────────────────────────────────────────

class SigmoidBoundaryTest : public ::testing::Test {};

TEST_F(SigmoidBoundaryTest, ZeroUpstreamGradGivesZeroInputGrad) {
    // dL/dx = 0 * σ(x)(1-σ(x)) = 0, regardless of the output value.
    auto out  = make_tensor({4}, {0.1f, 0.5f, 0.9f, 0.99f});
    auto grad = Tensor::zeros(make_shape({4}), 1);
    auto gx   = SigmoidOp::backward(grad, out);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(gx(i), 0.f);
}

TEST_F(SigmoidBoundaryTest, VeryNegativeInputDoesNotUnderflow) {
    // exp(-(-1000)) overflows to inf without numerical care.
    // sigmoid(-1000) should be finite and very close to 0.
    auto x   = make_tensor({1}, {-1000.f});
    auto out = SigmoidOp::forward(x);
    EXPECT_TRUE(std::isfinite(out(0)));
    EXPECT_NEAR(out(0), 0.f, 1e-4f);
}
