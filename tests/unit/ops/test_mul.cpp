#include <gtest/gtest.h>
#include <cstddef>

#include "ops/mul.h"   // pulls in autograd.h → tensor.h

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

static Shape make_shape(std::initializer_list<size_t> dims) {
    Shape s{};
    size_t i = 0;
    for (size_t d : dims) s[i++] = d;
    return s;
}

// ─────────────────────────────────────────────
// MulOp::forward — pure computation, no autograd
// ─────────────────────────────────────────────

class MulOpForwardTest : public ::testing::Test {};

TEST_F(MulOpForwardTest, ElementWise1D) {
    auto s = make_shape({4});
    auto a = Tensor::zeros(s, 1);
    auto b = Tensor::zeros(s, 1);
    for (size_t i = 0; i < 4; ++i) a.storage->data[i] = static_cast<float>(i + 1);  // 1 2 3 4
    for (size_t i = 0; i < 4; ++i) b.storage->data[i] = 2.0f;

    auto out = MulOp::forward(a, b);

    EXPECT_EQ(out.numel, 4u);
    EXPECT_FLOAT_EQ(out.storage->data[0], 2.0f);
    EXPECT_FLOAT_EQ(out.storage->data[1], 4.0f);
    EXPECT_FLOAT_EQ(out.storage->data[2], 6.0f);
    EXPECT_FLOAT_EQ(out.storage->data[3], 8.0f);
}

TEST_F(MulOpForwardTest, ElementWise2D) {
    auto s = make_shape({2, 3});
    auto a = Tensor::zeros(s, 2);
    auto b = Tensor::zeros(s, 2);
    for (size_t i = 0; i < 6; ++i) a.storage->data[i] = static_cast<float>(i);
    for (size_t i = 0; i < 6; ++i) b.storage->data[i] = static_cast<float>(i) * 2.0f;

    auto out = MulOp::forward(a, b);

    for (size_t i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(out.storage->data[i], static_cast<float>(i) * static_cast<float>(i) * 2.0f);
}

TEST_F(MulOpForwardTest, OutputShapeMatchesInput) {
    auto s = make_shape({3, 4});
    auto out = MulOp::forward(Tensor::ones(s, 2), Tensor::ones(s, 2));
    EXPECT_EQ(out.ndim, 2u);
    EXPECT_EQ(out.shape_at(0), 3u);
    EXPECT_EQ(out.shape_at(1), 4u);
}

TEST_F(MulOpForwardTest, OnesTimesOnesIsOnes) {
    auto s = make_shape({4, 4});
    auto out = MulOp::forward(Tensor::ones(s, 2), Tensor::ones(s, 2));
    for (size_t i = 0; i < out.numel; ++i)
        EXPECT_FLOAT_EQ(out.storage->data[i], 1.0f);
}

TEST_F(MulOpForwardTest, ZeroTimesAnythingIsZero) {
    auto s = make_shape({4});
    auto a = Tensor::zeros(s, 1);
    auto b = Tensor::ones(s, 1);
    for (size_t i = 0; i < 4; ++i) b.storage->data[i] = static_cast<float>(i * 100);
    auto out = MulOp::forward(a, b);
    for (size_t i = 0; i < out.numel; ++i)
        EXPECT_FLOAT_EQ(out.storage->data[i], 0.0f);
}

TEST_F(MulOpForwardTest, ShapeMismatchNdimAsserts) {
    EXPECT_DEATH(MulOp::forward(Tensor::ones(make_shape({4}), 1),
                                Tensor::ones(make_shape({2, 2}), 2)), "");
}

TEST_F(MulOpForwardTest, ShapeMismatchDimAsserts) {
    EXPECT_DEATH(MulOp::forward(Tensor::ones(make_shape({4}), 1),
                                Tensor::ones(make_shape({3}), 1)), "");
}

TEST_F(MulOpForwardTest, AlwaysProducesNoMeta) {
    // forward is pure computation — autograd wiring is mul()'s responsibility.
    auto s = make_shape({4});
    auto out = MulOp::forward(Tensor::ones(s, 1, /*requires_grad=*/true),
                              Tensor::ones(s, 1, /*requires_grad=*/true));
    EXPECT_EQ(out.autograd_meta, nullptr);
}

// ─────────────────────────────────────────────
// MulOp::backward — pure gradient computation
// ─────────────────────────────────────────────

class MulOpBackwardTest : public ::testing::Test {};

TEST_F(MulOpBackwardTest, GradAEqualsGradTimesB) {
    // With unit upstream gradient: dL/da = 1 * b = b
    auto s = make_shape({4});
    auto a = Tensor::zeros(s, 1);
    auto b = Tensor::zeros(s, 1);
    for (size_t i = 0; i < 4; ++i) { a.storage->data[i] = static_cast<float>(i + 1);
                                      b.storage->data[i] = static_cast<float>((i + 1) * 10); }

    auto grads = MulOp::backward(Tensor::ones(s, 1), a, b);

    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(grads[0].storage->data[i], b.storage->data[i]);
}

TEST_F(MulOpBackwardTest, GradBEqualsGradTimesA) {
    // With unit upstream gradient: dL/db = 1 * a = a
    auto s = make_shape({4});
    auto a = Tensor::zeros(s, 1);
    auto b = Tensor::zeros(s, 1);
    for (size_t i = 0; i < 4; ++i) { a.storage->data[i] = static_cast<float>(i + 1);
                                      b.storage->data[i] = static_cast<float>((i + 1) * 10); }

    auto grads = MulOp::backward(Tensor::ones(s, 1), a, b);

    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(grads[1].storage->data[i], a.storage->data[i]);
}

TEST_F(MulOpBackwardTest, NonUnitGradScales) {
    // grad_a[i] = grad[i] * b[i],  grad_b[i] = grad[i] * a[i]
    auto s = make_shape({3});
    auto a    = Tensor::zeros(s, 1);
    auto b    = Tensor::zeros(s, 1);
    auto grad = Tensor::zeros(s, 1);
    a.storage->data[0] = 1; a.storage->data[1] = 2; a.storage->data[2] = 3;
    b.storage->data[0] = 4; b.storage->data[1] = 5; b.storage->data[2] = 6;
    grad.storage->data[0] = 2; grad.storage->data[1] = 3; grad.storage->data[2] = 4;

    auto grads = MulOp::backward(grad, a, b);

    EXPECT_FLOAT_EQ(grads[0].storage->data[0], 2 * 4);
    EXPECT_FLOAT_EQ(grads[0].storage->data[1], 3 * 5);
    EXPECT_FLOAT_EQ(grads[0].storage->data[2], 4 * 6);
    EXPECT_FLOAT_EQ(grads[1].storage->data[0], 2 * 1);
    EXPECT_FLOAT_EQ(grads[1].storage->data[1], 3 * 2);
    EXPECT_FLOAT_EQ(grads[1].storage->data[2], 4 * 3);
}

TEST_F(MulOpBackwardTest, OutputShapeMatchesInput) {
    auto s    = make_shape({2, 3});
    auto grads = MulOp::backward(Tensor::ones(s, 2),
                                 Tensor::ones(s, 2),
                                 Tensor::ones(s, 2));
    EXPECT_EQ(grads[0].ndim, 2u);  EXPECT_EQ(grads[0].shape_at(0), 2u);  EXPECT_EQ(grads[0].shape_at(1), 3u);
    EXPECT_EQ(grads[1].ndim, 2u);  EXPECT_EQ(grads[1].shape_at(0), 2u);  EXPECT_EQ(grads[1].shape_at(1), 3u);
}

// ─────────────────────────────────────────────
// mul() — orchestration + autograd wiring
// ─────────────────────────────────────────────

class MulFuncTest : public ::testing::Test {};

TEST_F(MulFuncTest, ProducesCorrectValues) {
    auto s = make_shape({4});
    auto a = Tensor::zeros(s, 1);
    auto b = Tensor::zeros(s, 1);
    for (size_t i = 0; i < 4; ++i) { a.storage->data[i] = static_cast<float>(i + 1); b.storage->data[i] = 2.0f; }

    auto out = mul(a, b);

    EXPECT_FLOAT_EQ(out.storage->data[0], 2.0f);
    EXPECT_FLOAT_EQ(out.storage->data[1], 4.0f);
    EXPECT_FLOAT_EQ(out.storage->data[2], 6.0f);
    EXPECT_FLOAT_EQ(out.storage->data[3], 8.0f);
}

TEST_F(MulFuncTest, NoRequiresGradProducesNoMeta) {
    auto s   = make_shape({4});
    auto out = mul(Tensor::ones(s, 1), Tensor::ones(s, 1));
    EXPECT_EQ(out.autograd_meta, nullptr);
    EXPECT_FALSE(out.requires_grad());
}

TEST_F(MulFuncTest, OneRequiresGradProducesMeta) {
    auto s   = make_shape({4});
    auto out = mul(Tensor::ones(s, 1, /*requires_grad=*/true), Tensor::ones(s, 1));
    EXPECT_TRUE(out.requires_grad());
}

// ─────────────────────────────────────────────
// End-to-end autograd via mul()
// ─────────────────────────────────────────────

class MulOpAutogradTest : public ::testing::Test {};

TEST_F(MulOpAutogradTest, GradAEqualsB_1D) {
    // z = mul(a, b), seed = ones → dL/da = b
    auto s = make_shape({4});
    auto a = Tensor::zeros(s, 1, /*requires_grad=*/true);
    auto b = Tensor::zeros(s, 1, /*requires_grad=*/true);
    for (size_t i = 0; i < 4; ++i) { a.storage->data[i] = static_cast<float>(i + 1); }
    for (size_t i = 0; i < 4; ++i) { b.storage->data[i] = static_cast<float>((i + 1) * 10); }

    auto z = mul(a, b);
    backward(z);

    ASSERT_TRUE(a.has_grad());
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(a.grad().storage->data[i], b.storage->data[i]);
}

TEST_F(MulOpAutogradTest, GradBEqualsA_1D) {
    auto s = make_shape({4});
    auto a = Tensor::zeros(s, 1, /*requires_grad=*/true);
    auto b = Tensor::zeros(s, 1, /*requires_grad=*/true);
    for (size_t i = 0; i < 4; ++i) { a.storage->data[i] = static_cast<float>(i + 1); }
    for (size_t i = 0; i < 4; ++i) { b.storage->data[i] = static_cast<float>((i + 1) * 10); }

    auto z = mul(a, b);
    backward(z);

    ASSERT_TRUE(b.has_grad());
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(b.grad().storage->data[i], a.storage->data[i]);
}

TEST_F(MulOpAutogradTest, GradAEqualsB_2D) {
    auto s = make_shape({2, 3});
    auto a = Tensor::zeros(s, 2, /*requires_grad=*/true);
    auto b = Tensor::zeros(s, 2, /*requires_grad=*/true);
    for (size_t i = 0; i < 6; ++i) { a.storage->data[i] = static_cast<float>(i + 1); }
    for (size_t i = 0; i < 6; ++i) { b.storage->data[i] = static_cast<float>((i + 1) * 3); }

    auto z = mul(a, b);
    backward(z);

    for (size_t i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(a.grad().storage->data[i], b.storage->data[i]);
}

TEST_F(MulOpAutogradTest, OnlyOneInputRequiresGrad) {
    auto s = make_shape({4});
    auto a = Tensor::zeros(s, 1, /*requires_grad=*/true);
    auto b = Tensor::zeros(s, 1);
    for (size_t i = 0; i < 4; ++i) { a.storage->data[i] = 2.0f; b.storage->data[i] = 5.0f; }

    auto z = mul(a, b);
    backward(z);

    ASSERT_TRUE(a.has_grad());
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(a.grad().storage->data[i], 5.0f);  // dL/da = 1 * b = 5

    EXPECT_FALSE(b.has_grad());
}

TEST_F(MulOpAutogradTest, ForwardValuesNotAffectedByBackward) {
    auto s = make_shape({3});
    auto a = Tensor::zeros(s, 1, /*requires_grad=*/true);
    auto b = Tensor::zeros(s, 1, /*requires_grad=*/true);
    a.storage->data[0] = 2.0f; a.storage->data[1] = 3.0f; a.storage->data[2] = 4.0f;
    b.storage->data[0] = 5.0f; b.storage->data[1] = 6.0f; b.storage->data[2] = 7.0f;

    auto z = mul(a, b);
    EXPECT_FLOAT_EQ(z.storage->data[0], 10.0f);
    EXPECT_FLOAT_EQ(z.storage->data[1], 18.0f);
    EXPECT_FLOAT_EQ(z.storage->data[2], 28.0f);

    backward(z);

    EXPECT_FLOAT_EQ(z.storage->data[0], 10.0f);
    EXPECT_FLOAT_EQ(z.storage->data[1], 18.0f);
    EXPECT_FLOAT_EQ(z.storage->data[2], 28.0f);
}
