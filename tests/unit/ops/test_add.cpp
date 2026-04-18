#include <gtest/gtest.h>
#include <cstddef>

#include "ops/add.h"   // pulls in autograd.h → tensor.h
#include "ops/mul.h"

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
// AddOp::forward — pure computation, no autograd
// ─────────────────────────────────────────────

class AddOpForwardTest : public ::testing::Test {};

TEST_F(AddOpForwardTest, ElementWise1D) {
    auto a = make_tensor({4}, {1, 2, 3, 4});
    auto b = make_tensor({4}, {10, 20, 30, 40});

    auto out = AddOp::forward(a, b);

    EXPECT_EQ(out.numel, 4u);
    EXPECT_FLOAT_EQ(out(0), 11.f);
    EXPECT_FLOAT_EQ(out(1), 22.f);
    EXPECT_FLOAT_EQ(out(2), 33.f);
    EXPECT_FLOAT_EQ(out(3), 44.f);
}

TEST_F(AddOpForwardTest, ElementWise2D) {
    auto a = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
    auto b = make_tensor({2, 3}, {10, 20, 30, 40, 50, 60});

    auto out = AddOp::forward(a, b);

    EXPECT_EQ(out.shape_at(0), 2u);
    EXPECT_EQ(out.shape_at(1), 3u);
    for (size_t row = 0; row < 2; ++row)
        for (size_t col = 0; col < 3; ++col)
            EXPECT_FLOAT_EQ(out(row, col), a(row, col) + b(row, col));
}

TEST_F(AddOpForwardTest, OutputShapeMatchesInput_NoBC) {
    auto out = AddOp::forward(Tensor::ones(make_shape({3, 4}), 2),
                              Tensor::ones(make_shape({3, 4}), 2));
    EXPECT_EQ(out.ndim, 2u);
    EXPECT_EQ(out.shape_at(0), 3u);
    EXPECT_EQ(out.shape_at(1), 4u);
}

TEST_F(AddOpForwardTest, BroadcastDim0_RowPlusBatch) {
    // a: [1, 3]  — a single row,  b: [4, 3]  — 4 rows
    // out[i, j] = a[0, j] + b[i, j]
    auto a = make_tensor({1, 3}, {1, 2, 3});
    auto b = make_tensor({4, 3}, {10, 20, 30,
                                   40, 50, 60,
                                   70, 80, 90,
                                  100,110,120});
    auto out = AddOp::forward(a, b);

    EXPECT_EQ(out.shape_at(0), 4u);
    EXPECT_EQ(out.shape_at(1), 3u);
    EXPECT_FLOAT_EQ(out(0, 0), 11.f);  // 1+10
    EXPECT_FLOAT_EQ(out(0, 1), 22.f);  // 2+20
    EXPECT_FLOAT_EQ(out(0, 2), 33.f);  // 3+30
    EXPECT_FLOAT_EQ(out(1, 0), 41.f);  // 1+40
    EXPECT_FLOAT_EQ(out(3, 2), 123.f); // 3+120
}

TEST_F(AddOpForwardTest, BroadcastDim0_Reversed) {
    // Swap a and b — commutativity of broadcasting
    auto a = make_tensor({4, 3}, {10, 20, 30,
                                   40, 50, 60,
                                   70, 80, 90,
                                  100,110,120});
    auto b = make_tensor({1, 3}, {1, 2, 3});
    auto out = AddOp::forward(a, b);

    EXPECT_EQ(out.shape_at(0), 4u);
    EXPECT_FLOAT_EQ(out(0, 0), 11.f);
    EXPECT_FLOAT_EQ(out(3, 2), 123.f);
}

TEST_F(AddOpForwardTest, BroadcastDim1_ColPlusBatch) {
    // a: [3, 4],  b: [3, 1] — column vector
    auto a = make_tensor({3, 4}, {1, 2, 3, 4,
                                   5, 6, 7, 8,
                                   9,10,11,12});
    auto b = make_tensor({3, 1}, {100, 200, 300});
    auto out = AddOp::forward(a, b);

    EXPECT_EQ(out.shape_at(0), 3u);
    EXPECT_EQ(out.shape_at(1), 4u);
    EXPECT_FLOAT_EQ(out(0, 0), 101.f);
    EXPECT_FLOAT_EQ(out(0, 3), 104.f);
    EXPECT_FLOAT_EQ(out(1, 0), 205.f);
    EXPECT_FLOAT_EQ(out(1, 3), 208.f);
    EXPECT_FLOAT_EQ(out(2, 2), 311.f);
}

TEST_F(AddOpForwardTest, BroadcastBothDims) {
    // a: [3, 1],  b: [1, 4]  →  out: [3, 4]
    auto a = make_tensor({3, 1}, {10, 20, 30});
    auto b = make_tensor({1, 4}, {1, 2, 3, 4});
    auto out = AddOp::forward(a, b);

    EXPECT_EQ(out.shape_at(0), 3u);
    EXPECT_EQ(out.shape_at(1), 4u);
    EXPECT_FLOAT_EQ(out(0, 0), 11.f);  // 10+1
    EXPECT_FLOAT_EQ(out(0, 3), 14.f);  // 10+4
    EXPECT_FLOAT_EQ(out(1, 0), 21.f);  // 20+1
    EXPECT_FLOAT_EQ(out(2, 3), 34.f);  // 30+4
}

TEST_F(AddOpForwardTest, NdimMismatchAsserts) {
    EXPECT_DEATH(AddOp::forward(Tensor::ones(make_shape({4}), 1),
                                Tensor::ones(make_shape({2, 2}), 2)), "");
}

TEST_F(AddOpForwardTest, IncompatibleDimAsserts) {
    // [3, 4] + [2, 4]: dim 0 is neither equal nor 1
    EXPECT_DEATH(AddOp::forward(Tensor::ones(make_shape({3, 4}), 2),
                                Tensor::ones(make_shape({2, 4}), 2)), "");
}

TEST_F(AddOpForwardTest, AlwaysProducesNoMeta) {
    // forward is pure computation — autograd wiring is add()'s responsibility.
    auto s = make_shape({4});
    auto out = AddOp::forward(Tensor::ones(s, 1, /*requires_grad=*/true),
                              Tensor::ones(s, 1, /*requires_grad=*/true));
    EXPECT_EQ(out.autograd_meta, nullptr);
}

// ─────────────────────────────────────────────
// AddOp::backward — pure gradient computation
// ─────────────────────────────────────────────

class AddOpBackwardTest : public ::testing::Test {};

TEST_F(AddOpBackwardTest, NoBroadcast_GradPassesThrough) {
    // dL/da = dL/db = grad  (no summation needed)
    auto a    = make_tensor({3}, {1, 2, 3});
    auto b    = make_tensor({3}, {4, 5, 6});
    auto grad = make_tensor({3}, {7, 8, 9});

    auto grads = AddOp::backward(grad, a, b);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(grads[0](i), grad(i));
        EXPECT_FLOAT_EQ(grads[1](i), grad(i));
    }
}

TEST_F(AddOpBackwardTest, BroadcastDim0_GradSummedForA) {
    // a: [1, 3],  b: [4, 3].  grad_a[0, j] = sum over i of grad[i, j]
    auto a = make_tensor({1, 3}, {0, 0, 0});
    auto b = Tensor::zeros(make_shape({4, 3}), 2);
    auto grad = Tensor::ones(make_shape({4, 3}), 2);

    auto grads = AddOp::backward(grad, a, b);

    // grad_a shape is [1, 3]; each element is sum of 4 ones = 4
    EXPECT_EQ(grads[0].shape_at(0), 1u);
    EXPECT_EQ(grads[0].shape_at(1), 3u);
    for (size_t j = 0; j < 3; ++j)
        EXPECT_FLOAT_EQ(grads[0](0, j), 4.f);

    // grad_b shape is [4, 3]; passes through unchanged
    EXPECT_EQ(grads[1].shape_at(0), 4u);
    for (size_t row = 0; row < 4; ++row)
        for (size_t col = 0; col < 3; ++col)
            EXPECT_FLOAT_EQ(grads[1](row, col), 1.f);
}

TEST_F(AddOpBackwardTest, BroadcastDim1_GradSummedForB) {
    // a: [3, 4],  b: [3, 1].  grad_b[i, 0] = sum over j of grad[i, j]
    auto a = Tensor::zeros(make_shape({3, 4}), 2);
    auto b = make_tensor({3, 1}, {0, 0, 0});
    auto grad = Tensor::ones(make_shape({3, 4}), 2);

    auto grads = AddOp::backward(grad, a, b);

    EXPECT_EQ(grads[1].shape_at(0), 3u);
    EXPECT_EQ(grads[1].shape_at(1), 1u);
    for (size_t i = 0; i < 3; ++i)
        EXPECT_FLOAT_EQ(grads[1](i, 0), 4.f);
}

TEST_F(AddOpBackwardTest, BroadcastBothDims_NonUnitGrad) {
    // a: [3, 1],  b: [1, 4].  grad: [3, 4]
    // grad_a[i, 0] = sum_j grad[i, j]
    // grad_b[0, j] = sum_i grad[i, j]
    auto a = make_tensor({3, 1}, {0, 0, 0});
    auto b = make_tensor({1, 4}, {0, 0, 0, 0});
    auto grad = Tensor::zeros(make_shape({3, 4}), 2);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            grad(i, j) = static_cast<float>((i + 1) * 10 + (j + 1));

    auto grads = AddOp::backward(grad, a, b);

    // i=0: 11+12+13+14 = 50
    EXPECT_FLOAT_EQ(grads[0](0, 0), 50.f);
    // i=1: 21+22+23+24 = 90
    EXPECT_FLOAT_EQ(grads[0](1, 0), 90.f);
    // i=2: 31+32+33+34 = 130
    EXPECT_FLOAT_EQ(grads[0](2, 0), 130.f);

    // j=0: 11+21+31 = 63
    EXPECT_FLOAT_EQ(grads[1](0, 0), 63.f);
    // j=3: 14+24+34 = 72
    EXPECT_FLOAT_EQ(grads[1](0, 3), 72.f);
}

TEST_F(AddOpBackwardTest, OutputShapeMatchesInputs) {
    auto a = Tensor::zeros(make_shape({2, 3}), 2);
    auto b = Tensor::zeros(make_shape({2, 3}), 2);
    auto grads = AddOp::backward(Tensor::ones(make_shape({2, 3}), 2), a, b);
    EXPECT_EQ(grads[0].shape_at(0), 2u); EXPECT_EQ(grads[0].shape_at(1), 3u);
    EXPECT_EQ(grads[1].shape_at(0), 2u); EXPECT_EQ(grads[1].shape_at(1), 3u);
}

// ─────────────────────────────────────────────
// add() — orchestration + autograd wiring
// ─────────────────────────────────────────────

class AddFuncTest : public ::testing::Test {};

TEST_F(AddFuncTest, ProducesCorrectValues_NoBC) {
    auto a = make_tensor({3}, {1, 2, 3});
    auto b = make_tensor({3}, {10, 20, 30});
    auto out = add(a, b);
    EXPECT_FLOAT_EQ(out(0), 11.f);
    EXPECT_FLOAT_EQ(out(1), 22.f);
    EXPECT_FLOAT_EQ(out(2), 33.f);
}

TEST_F(AddFuncTest, ProducesCorrectValues_Broadcast) {
    auto a = make_tensor({1, 3}, {1, 2, 3});
    auto b = make_tensor({2, 3}, {10, 20, 30, 40, 50, 60});
    auto out = add(a, b);
    EXPECT_FLOAT_EQ(out(0, 0), 11.f);
    EXPECT_FLOAT_EQ(out(1, 2), 63.f);
}

TEST_F(AddFuncTest, NoRequiresGradProducesNoMeta) {
    auto s   = make_shape({4});
    auto out = add(Tensor::ones(s, 1), Tensor::ones(s, 1));
    EXPECT_EQ(out.autograd_meta, nullptr);
    EXPECT_FALSE(out.requires_grad());
}

TEST_F(AddFuncTest, OneRequiresGradProducesMeta) {
    auto s   = make_shape({4});
    auto out = add(Tensor::ones(s, 1, /*requires_grad=*/true), Tensor::ones(s, 1));
    EXPECT_TRUE(out.requires_grad());
}

// ─────────────────────────────────────────────
// End-to-end autograd via add()
// ─────────────────────────────────────────────

class AddAutogradTest : public ::testing::Test {};

TEST_F(AddAutogradTest, NoBroadcast_GradIsOne) {
    // z = add(a, b), seed=ones → dL/da = dL/db = ones
    auto a = make_tensor({4}, {1, 2, 3, 4}, /*requires_grad=*/true);
    auto b = make_tensor({4}, {5, 6, 7, 8}, /*requires_grad=*/true);

    auto z = add(a, b);
    backward(z);

    ASSERT_TRUE(a.has_grad());
    ASSERT_TRUE(b.has_grad());
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(a.grad()(i), 1.f);
        EXPECT_FLOAT_EQ(b.grad()(i), 1.f);
    }
}

TEST_F(AddAutogradTest, BroadcastDim0_GradSummedForA) {
    // a: [1, 3] requires_grad,  b: [4, 3] requires_grad
    // dL/da[0, j] = sum_i dL/dout[i, j] = 4 (with ones seed)
    auto a = make_tensor({1, 3}, {0, 0, 0}, /*requires_grad=*/true);
    auto b = Tensor::zeros(make_shape({4, 3}), 2, /*requires_grad=*/true);

    auto z = add(a, b);
    backward(z);

    ASSERT_TRUE(a.has_grad());
    for (size_t j = 0; j < 3; ++j)
        EXPECT_FLOAT_EQ(a.grad()(0, j), 4.f);

    ASSERT_TRUE(b.has_grad());
    for (size_t row = 0; row < 4; ++row)
        for (size_t col = 0; col < 3; ++col)
            EXPECT_FLOAT_EQ(b.grad()(row, col), 1.f);
}

TEST_F(AddAutogradTest, OnlyARequiresGrad) {
    auto a = make_tensor({4}, {1, 2, 3, 4}, /*requires_grad=*/true);
    auto b = make_tensor({4}, {5, 6, 7, 8});

    auto z = add(a, b);
    backward(z);

    ASSERT_TRUE(a.has_grad());
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(a.grad()(i), 1.f);

    EXPECT_FALSE(b.has_grad());
}

TEST_F(AddAutogradTest, ChainWithMul) {
    // z = add(a, b) -> w = mul(z, c)
    // dL/da = dL/db = c  (via chain rule: dL/dz = c, dL/da = dL/dz * 1 = c)
    auto a = make_tensor({3}, {1, 2, 3}, /*requires_grad=*/true);
    auto b = make_tensor({3}, {4, 5, 6}, /*requires_grad=*/true);
    auto c = make_tensor({3}, {2, 3, 4});

    auto z = add(a, b);
    auto w = mul(z, c);
    backward(w);

    ASSERT_TRUE(a.has_grad());
    ASSERT_TRUE(b.has_grad());
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(a.grad()(i), c(i));
        EXPECT_FLOAT_EQ(b.grad()(i), c(i));
    }
}
