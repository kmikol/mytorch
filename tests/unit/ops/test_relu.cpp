#include <gtest/gtest.h>
#include <cstddef>

#include "ops/activations/relu.h"

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
// ReLUOp::forward — pure computation
// ─────────────────────────────────────────────

class ReLUOpForwardTest : public ::testing::Test {};

TEST_F(ReLUOpForwardTest, PositivePassThrough) {
    auto x   = make_tensor({4}, {1.f, 2.f, 3.f, 4.f});
    auto out = ReLUOp::forward(x);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(out(i), x(i));
}

TEST_F(ReLUOpForwardTest, NegativeClampedToZero) {
    auto x   = make_tensor({4}, {-1.f, -2.f, -3.f, -4.f});
    auto out = ReLUOp::forward(x);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(out(i), 0.f);
}

TEST_F(ReLUOpForwardTest, MixedSigns) {
    auto x   = make_tensor({6}, {-3.f, -1.f, 0.f, 1.f, 2.f, 5.f});
    auto out = ReLUOp::forward(x);
    EXPECT_FLOAT_EQ(out(0), 0.f);
    EXPECT_FLOAT_EQ(out(1), 0.f);
    EXPECT_FLOAT_EQ(out(2), 0.f);  // exactly zero stays zero
    EXPECT_FLOAT_EQ(out(3), 1.f);
    EXPECT_FLOAT_EQ(out(4), 2.f);
    EXPECT_FLOAT_EQ(out(5), 5.f);
}

TEST_F(ReLUOpForwardTest, OutputShapePreserved) {
    auto out = ReLUOp::forward(Tensor::zeros(make_shape({3, 4}), 2));
    EXPECT_EQ(out.ndim, 2u);
    EXPECT_EQ(out.shape_at(0), 3u);
    EXPECT_EQ(out.shape_at(1), 4u);
}

TEST_F(ReLUOpForwardTest, AlwaysProducesNoMeta) {
    auto x   = make_tensor({4}, {1.f, 2.f, 3.f, 4.f}, /*requires_grad=*/true);
    auto out = ReLUOp::forward(x);
    EXPECT_EQ(out.autograd_meta, nullptr);
}

// ─────────────────────────────────────────────
// ReLUOp::backward — pure gradient computation
// ─────────────────────────────────────────────

class ReLUOpBackwardTest : public ::testing::Test {};

TEST_F(ReLUOpBackwardTest, PositiveInputPassesGrad) {
    auto x    = make_tensor({4}, {1.f, 2.f, 3.f, 4.f});
    auto grad = make_tensor({4}, {5.f, 6.f, 7.f, 8.f});
    auto gx   = ReLUOp::backward(grad, x);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(gx(i), grad(i));
}

TEST_F(ReLUOpBackwardTest, NegativeInputZerosGrad) {
    auto x    = make_tensor({4}, {-1.f, -2.f, -3.f, -4.f});
    auto grad = make_tensor({4}, {5.f, 6.f, 7.f, 8.f});
    auto gx   = ReLUOp::backward(grad, x);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(gx(i), 0.f);
}

TEST_F(ReLUOpBackwardTest, MixedMask) {
    auto x    = make_tensor({4}, {-1.f, 2.f, -3.f, 4.f});
    auto grad = make_tensor({4}, {1.f, 1.f,  1.f, 1.f});
    auto gx   = ReLUOp::backward(grad, x);
    EXPECT_FLOAT_EQ(gx(0), 0.f);
    EXPECT_FLOAT_EQ(gx(1), 1.f);
    EXPECT_FLOAT_EQ(gx(2), 0.f);
    EXPECT_FLOAT_EQ(gx(3), 1.f);
}

TEST_F(ReLUOpBackwardTest, GradScaledByUpstream) {
    auto x    = make_tensor({3}, {1.f, -1.f, 2.f});
    auto grad = make_tensor({3}, {3.f,  5.f, 7.f});
    auto gx   = ReLUOp::backward(grad, x);
    EXPECT_FLOAT_EQ(gx(0), 3.f);   // positive: pass grad
    EXPECT_FLOAT_EQ(gx(1), 0.f);   // negative: zero
    EXPECT_FLOAT_EQ(gx(2), 7.f);   // positive: pass grad
}

// ─────────────────────────────────────────────
// relu() — orchestration + autograd wiring
// ─────────────────────────────────────────────

class ReLUFuncTest : public ::testing::Test {};

TEST_F(ReLUFuncTest, ProducesCorrectValues) {
    auto x   = make_tensor({4}, {-2.f, -1.f, 1.f, 2.f});
    auto out = relu(x);
    EXPECT_FLOAT_EQ(out(0), 0.f);
    EXPECT_FLOAT_EQ(out(1), 0.f);
    EXPECT_FLOAT_EQ(out(2), 1.f);
    EXPECT_FLOAT_EQ(out(3), 2.f);
}

TEST_F(ReLUFuncTest, NoRequiresGradProducesNoMeta) {
    auto out = relu(make_tensor({4}, {1.f, 2.f, 3.f, 4.f}));
    EXPECT_EQ(out.autograd_meta, nullptr);
}

TEST_F(ReLUFuncTest, RequiresGradProducesMeta) {
    auto out = relu(make_tensor({4}, {1.f, 2.f, 3.f, 4.f}, /*requires_grad=*/true));
    EXPECT_TRUE(out.requires_grad());
}

// ─────────────────────────────────────────────
// End-to-end autograd via relu()
// ─────────────────────────────────────────────

class ReLUAutogradTest : public ::testing::Test {};

TEST_F(ReLUAutogradTest, GradMaskedBySign) {
    // z = relu(x), seed=ones → dL/dx[i] = 1 if x[i]>0, 0 otherwise
    auto x = make_tensor({4}, {-2.f, -1.f, 1.f, 2.f}, /*requires_grad=*/true);
    auto z = relu(x);
    backward(z);

    ASSERT_TRUE(x.has_grad());
    EXPECT_FLOAT_EQ(x.grad()(0), 0.f);
    EXPECT_FLOAT_EQ(x.grad()(1), 0.f);
    EXPECT_FLOAT_EQ(x.grad()(2), 1.f);
    EXPECT_FLOAT_EQ(x.grad()(3), 1.f);
}

TEST_F(ReLUAutogradTest, AllNegativeGivesZeroGrad) {
    auto x = make_tensor({3}, {-3.f, -1.f, -0.5f}, /*requires_grad=*/true);
    auto z = relu(x);
    backward(z);

    ASSERT_TRUE(x.has_grad());
    for (size_t i = 0; i < 3; ++i)
        EXPECT_FLOAT_EQ(x.grad()(i), 0.f);
}

TEST_F(ReLUAutogradTest, ForwardValuesUnchangedByBackward) {
    auto x = make_tensor({3}, {-1.f, 0.f, 2.f}, /*requires_grad=*/true);
    auto z = relu(x);
    EXPECT_FLOAT_EQ(z(0), 0.f);
    EXPECT_FLOAT_EQ(z(1), 0.f);
    EXPECT_FLOAT_EQ(z(2), 2.f);
    backward(z);
    EXPECT_FLOAT_EQ(z(0), 0.f);
    EXPECT_FLOAT_EQ(z(2), 2.f);
}
