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
    for (size_t i = 0; i < 4; ++i) a(i) = static_cast<float>(i + 1);  // 1 2 3 4
    for (size_t i = 0; i < 4; ++i) b(i) = 2.0f;

    auto out = MulOp::forward(a, b);

    EXPECT_EQ(out.numel, 4u);
    EXPECT_FLOAT_EQ(out(0), 2.0f);
    EXPECT_FLOAT_EQ(out(1), 4.0f);
    EXPECT_FLOAT_EQ(out(2), 6.0f);
    EXPECT_FLOAT_EQ(out(3), 8.0f);
}

TEST_F(MulOpForwardTest, ElementWise2D) {
    auto s = make_shape({2, 3});
    auto a = Tensor::zeros(s, 2);
    auto b = Tensor::zeros(s, 2);
    for (size_t row = 0; row < 2; ++row)
        for (size_t col = 0; col < 3; ++col) {
            size_t i = row * 3 + col;
            a(row, col) = static_cast<float>(i);
            b(row, col) = static_cast<float>(i) * 2.0f;
        }

    auto out = MulOp::forward(a, b);

    for (size_t row = 0; row < 2; ++row)
        for (size_t col = 0; col < 3; ++col) {
            size_t i = row * 3 + col;
            EXPECT_FLOAT_EQ(out(row, col), static_cast<float>(i) * static_cast<float>(i) * 2.0f);
        }
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
    for (size_t row = 0; row < 4; ++row)
        for (size_t col = 0; col < 4; ++col)
            EXPECT_FLOAT_EQ(out(row, col), 1.0f);
}

TEST_F(MulOpForwardTest, ZeroTimesAnythingIsZero) {
    auto s = make_shape({4});
    auto a = Tensor::zeros(s, 1);
    auto b = Tensor::ones(s, 1);
    for (size_t i = 0; i < 4; ++i) b(i) = static_cast<float>(i * 100);
    auto out = MulOp::forward(a, b);
    for (size_t i = 0; i < out.numel; ++i)
        EXPECT_FLOAT_EQ(out(i), 0.0f);
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
    for (size_t i = 0; i < 4; ++i) { a(i) = static_cast<float>(i + 1);
                                      b(i) = static_cast<float>((i + 1) * 10); }

    auto grads = MulOp::backward(Tensor::ones(s, 1), a, b);

    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(grads[0](i), b(i));
}

TEST_F(MulOpBackwardTest, GradBEqualsGradTimesA) {
    // With unit upstream gradient: dL/db = 1 * a = a
    auto s = make_shape({4});
    auto a = Tensor::zeros(s, 1);
    auto b = Tensor::zeros(s, 1);
    for (size_t i = 0; i < 4; ++i) { a(i) = static_cast<float>(i + 1);
                                      b(i) = static_cast<float>((i + 1) * 10); }

    auto grads = MulOp::backward(Tensor::ones(s, 1), a, b);

    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(grads[1](i), a(i));
}

TEST_F(MulOpBackwardTest, NonUnitGradScales) {
    // grad_a[i] = grad[i] * b[i],  grad_b[i] = grad[i] * a[i]
    auto s = make_shape({3});
    auto a    = Tensor::zeros(s, 1);
    auto b    = Tensor::zeros(s, 1);
    auto grad = Tensor::zeros(s, 1);
    a(0) = 1; a(1) = 2; a(2) = 3;
    b(0) = 4; b(1) = 5; b(2) = 6;
    grad(0) = 2; grad(1) = 3; grad(2) = 4;

    auto grads = MulOp::backward(grad, a, b);

    EXPECT_FLOAT_EQ(grads[0](0), 2 * 4);
    EXPECT_FLOAT_EQ(grads[0](1), 3 * 5);
    EXPECT_FLOAT_EQ(grads[0](2), 4 * 6);
    EXPECT_FLOAT_EQ(grads[1](0), 2 * 1);
    EXPECT_FLOAT_EQ(grads[1](1), 3 * 2);
    EXPECT_FLOAT_EQ(grads[1](2), 4 * 3);
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
    for (size_t i = 0; i < 4; ++i) { a(i) = static_cast<float>(i + 1); b(i) = 2.0f; }

    auto out = mul(a, b);

    EXPECT_FLOAT_EQ(out(0), 2.0f);
    EXPECT_FLOAT_EQ(out(1), 4.0f);
    EXPECT_FLOAT_EQ(out(2), 6.0f);
    EXPECT_FLOAT_EQ(out(3), 8.0f);
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
    for (size_t i = 0; i < 4; ++i) { a(i) = static_cast<float>(i + 1); }
    for (size_t i = 0; i < 4; ++i) { b(i) = static_cast<float>((i + 1) * 10); }

    auto z = mul(a, b);
    backward(z);

    ASSERT_TRUE(a.has_grad());
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(a.grad()(i), b(i));
}

TEST_F(MulOpAutogradTest, GradBEqualsA_1D) {
    auto s = make_shape({4});
    auto a = Tensor::zeros(s, 1, /*requires_grad=*/true);
    auto b = Tensor::zeros(s, 1, /*requires_grad=*/true);
    for (size_t i = 0; i < 4; ++i) { a(i) = static_cast<float>(i + 1); }
    for (size_t i = 0; i < 4; ++i) { b(i) = static_cast<float>((i + 1) * 10); }

    auto z = mul(a, b);
    backward(z);

    ASSERT_TRUE(b.has_grad());
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(b.grad()(i), a(i));
}

TEST_F(MulOpAutogradTest, GradAEqualsB_2D) {
    auto s = make_shape({2, 3});
    auto a = Tensor::zeros(s, 2, /*requires_grad=*/true);
    auto b = Tensor::zeros(s, 2, /*requires_grad=*/true);
    for (size_t row = 0; row < 2; ++row)
        for (size_t col = 0; col < 3; ++col) {
            size_t i = row * 3 + col;
            a(row, col) = static_cast<float>(i + 1);
            b(row, col) = static_cast<float>((i + 1) * 3);
        }

    auto z = mul(a, b);
    backward(z);

    for (size_t row = 0; row < 2; ++row)
        for (size_t col = 0; col < 3; ++col)
            EXPECT_FLOAT_EQ(a.grad()(row, col), b(row, col));
}

TEST_F(MulOpAutogradTest, OnlyOneInputRequiresGrad) {
    auto s = make_shape({4});
    auto a = Tensor::zeros(s, 1, /*requires_grad=*/true);
    auto b = Tensor::zeros(s, 1);
    for (size_t i = 0; i < 4; ++i) { a(i) = 2.0f; b(i) = 5.0f; }

    auto z = mul(a, b);
    backward(z);

    ASSERT_TRUE(a.has_grad());
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(a.grad()(i), 5.0f);  // dL/da = 1 * b = 5

    EXPECT_FALSE(b.has_grad());
}

// ─────────────────────────────────────────────
// Non-contiguous inputs (stride-aware paths)
// ─────────────────────────────────────────────

class MulOpNonContiguousTest : public ::testing::Test {};

// Transposing a [2,3] tensor produces a non-contiguous [2,3] view.
// MulOp::forward must read via actual strides, not a flat offset assumption.
TEST_F(MulOpNonContiguousTest, ForwardWithTransposedInput) {
    // A  = [[1,2,3],[4,5,6]]  shape [2,3]  (contiguous)
    // B  = [[10,20],[30,40],[50,60]]  shape [3,2]  (contiguous)
    // B.T() = [[10,30,50],[20,40,60]]  shape [2,3]  (non-contiguous)
    // A * B.T() = [[10,60,150],[80,200,360]]
    auto a = Tensor::zeros(make_shape({2, 3}), 2);
    a(0,0)=1; a(0,1)=2; a(0,2)=3;
    a(1,0)=4; a(1,1)=5; a(1,2)=6;

    auto b_row = Tensor::zeros(make_shape({3, 2}), 2);
    b_row(0,0)=10; b_row(0,1)=20;
    b_row(1,0)=30; b_row(1,1)=40;
    b_row(2,0)=50; b_row(2,1)=60;
    auto b = b_row.T();  // non-contiguous [2,3]

    ASSERT_FALSE(b.is_contiguous());

    auto out = MulOp::forward(a, b);

    EXPECT_FLOAT_EQ(out(0,0), 10.f);
    EXPECT_FLOAT_EQ(out(0,1), 60.f);
    EXPECT_FLOAT_EQ(out(0,2), 150.f);
    EXPECT_FLOAT_EQ(out(1,0), 80.f);
    EXPECT_FLOAT_EQ(out(1,1), 200.f);
    EXPECT_FLOAT_EQ(out(1,2), 360.f);
}

// backward must also apply the correct strides when a_save / b_save are non-contiguous.
TEST_F(MulOpNonContiguousTest, BackwardWithTransposedSavedInput) {
    auto a_row = Tensor::zeros(make_shape({3, 2}), 2);
    a_row(0,0)=2; a_row(0,1)=3;
    a_row(1,0)=4; a_row(1,1)=5;
    a_row(2,0)=6; a_row(2,1)=7;
    auto a = a_row.T();  // non-contiguous [2,3]: a[i,j] = a_row[j,i]

    auto b = Tensor::zeros(make_shape({2, 3}), 2);
    b(0,0)=1; b(0,1)=1; b(0,2)=1;
    b(1,0)=1; b(1,1)=1; b(1,2)=1;

    auto grad = Tensor::ones(make_shape({2, 3}), 2);

    // grad_a = grad * b = ones * ones = a (since b=ones, grad_a = a's values)
    // grad_b = grad * a_transposed_values
    auto [grad_a, grad_b] = [&] {
        auto v = MulOp::backward(grad, a, b);
        return std::make_pair(v[0], v[1]);
    }();

    // a(0,0)=a_row(0,0)=2, a(0,1)=a_row(1,0)=4, a(0,2)=a_row(2,0)=6
    // a(1,0)=a_row(0,1)=3, a(1,1)=a_row(1,1)=5, a(1,2)=a_row(2,1)=7
    EXPECT_FLOAT_EQ(grad_b(0,0), 2.f);
    EXPECT_FLOAT_EQ(grad_b(0,1), 4.f);
    EXPECT_FLOAT_EQ(grad_b(0,2), 6.f);
    EXPECT_FLOAT_EQ(grad_b(1,0), 3.f);
    EXPECT_FLOAT_EQ(grad_b(1,1), 5.f);
    EXPECT_FLOAT_EQ(grad_b(1,2), 7.f);
}

TEST_F(MulOpAutogradTest, ForwardValuesNotAffectedByBackward) {
    auto s = make_shape({3});
    auto a = Tensor::zeros(s, 1, /*requires_grad=*/true);
    auto b = Tensor::zeros(s, 1, /*requires_grad=*/true);
    a(0) = 2.0f; a(1) = 3.0f; a(2) = 4.0f;
    b(0) = 5.0f; b(1) = 6.0f; b(2) = 7.0f;

    auto z = mul(a, b);
    EXPECT_FLOAT_EQ(z(0), 10.0f);
    EXPECT_FLOAT_EQ(z(1), 18.0f);
    EXPECT_FLOAT_EQ(z(2), 28.0f);

    backward(z);

    EXPECT_FLOAT_EQ(z(0), 10.0f);
    EXPECT_FLOAT_EQ(z(1), 18.0f);
    EXPECT_FLOAT_EQ(z(2), 28.0f);
}
