// tests/unit/ops/test_matmul.cpp
//
// Unit tests for matmul() and MatMulOp.
//
// Test organisation:
//   1. MatMulOpForwardTest    — forward() in isolation: shapes and values
//   2. MatMulTransposedInput  — forward() with non-contiguous (T()) inputs
//   3. MatMulOpBackwardTest   — backward() gradients in isolation
//   4. MatMulFuncTest         — matmul() autograd wiring (requires_grad)
//   5. MatMulAutogradTest     — full backward() pass through matmul()

#include <gtest/gtest.h>
#include <cstddef>

#include "ops/matmul.h"

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

static Shape make_shape(std::initializer_list<size_t> dims) {
    Shape s{};
    size_t i = 0;
    for (size_t d : dims) s[i++] = d;
    return s;
}

// Build a tensor from a flat data list and dimension sizes.
static Tensor make_tensor(std::initializer_list<float>  data,
                          std::initializer_list<size_t> dims,
                          bool requires_grad = false) {
    Shape  s{};
    size_t ndim = 0;
    for (size_t d : dims) s[ndim++] = d;

    Tensor t(s, ndim, requires_grad);
    size_t i = 0;
    for (float v : data) t.storage->data[i++] = v;
    return t;
}

// ════════════════════════════════════════════════════════════════════════════
// 1. MatMulOpForwardTest — shape and value correctness
// ════════════════════════════════════════════════════════════════════════════

class MatMulOpForwardTest : public ::testing::Test {};

TEST_F(MatMulOpForwardTest, OutputShapeRectangular) {
    // [2,3] @ [3,4] → [2,4]
    auto out = MatMulOp::forward(Tensor::zeros(make_shape({2,3}), 2),
                                 Tensor::zeros(make_shape({3,4}), 2));
    EXPECT_EQ(out.shape_at(0), 2u);
    EXPECT_EQ(out.shape_at(1), 4u);
}

TEST_F(MatMulOpForwardTest, OutputShapeSquare) {
    auto out = MatMulOp::forward(Tensor::zeros(make_shape({3,3}), 2),
                                 Tensor::zeros(make_shape({3,3}), 2));
    EXPECT_EQ(out.shape_at(0), 3u);
    EXPECT_EQ(out.shape_at(1), 3u);
}

TEST_F(MatMulOpForwardTest, OutputShapeColumnVector) {
    // [2,3] @ [3,1] → [2,1]
    auto out = MatMulOp::forward(Tensor::zeros(make_shape({2,3}), 2),
                                 Tensor::zeros(make_shape({3,1}), 2));
    EXPECT_EQ(out.shape_at(0), 2u);
    EXPECT_EQ(out.shape_at(1), 1u);
}

TEST_F(MatMulOpForwardTest, OutputNdimIsTwo) {
    auto out = MatMulOp::forward(Tensor::zeros(make_shape({2,3}), 2),
                                 Tensor::zeros(make_shape({3,4}), 2));
    EXPECT_EQ(out.ndim, 2u);
}

TEST_F(MatMulOpForwardTest, OutputIsContiguous) {
    auto out = MatMulOp::forward(make_tensor({1,2,3,4,5,6}, {2,3}),
                                 make_tensor({1,2,3,4,5,6}, {3,2}));
    EXPECT_TRUE(out.is_contiguous());
}

TEST_F(MatMulOpForwardTest, AlwaysProducesNoMeta) {
    // forward is pure computation — autograd wiring is matmul()'s responsibility.
    auto A = make_tensor({1,2,3,4}, {2,2}, /*requires_grad=*/true);
    auto B = make_tensor({1,2,3,4}, {2,2}, /*requires_grad=*/true);
    auto out = MatMulOp::forward(A, B);
    EXPECT_EQ(out.autograd_meta, nullptr);
}

TEST_F(MatMulOpForwardTest, Compute2x3by3x2) {
    // A = [[1,2,3],[4,5,6]]   B = [[7,8],[9,10],[11,12]]
    // C[0,0] = 1*7  + 2*9  + 3*11 = 58
    // C[0,1] = 1*8  + 2*10 + 3*12 = 64
    // C[1,0] = 4*7  + 5*9  + 6*11 = 139
    // C[1,1] = 4*8  + 5*10 + 6*12 = 154
    auto A = make_tensor({1,2,3,4,5,6},    {2,3});
    auto B = make_tensor({7,8,9,10,11,12}, {3,2});
    auto C = MatMulOp::forward(A, B);

    EXPECT_FLOAT_EQ(C(0,0),  58.f);
    EXPECT_FLOAT_EQ(C(0,1),  64.f);
    EXPECT_FLOAT_EQ(C(1,0), 139.f);
    EXPECT_FLOAT_EQ(C(1,1), 154.f);
}

TEST_F(MatMulOpForwardTest, MultiplyByIdentity) {
    auto I = make_tensor({1,0,0, 0,1,0, 0,0,1}, {3,3});
    auto X = make_tensor({1,2,3, 4,5,6, 7,8,9}, {3,3});
    auto R = MatMulOp::forward(X, I);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(R(i,j), X(i,j)) << "mismatch at [" << i << "," << j << "]";
}

TEST_F(MatMulOpForwardTest, MultiplyByZero) {
    auto A = make_tensor({1,2,3,4,5,6}, {2,3});
    auto Z = Tensor::zeros(make_shape({3,2}), 2);
    auto R = MatMulOp::forward(A, Z);
    for (size_t row = 0; row < 2; ++row)
        for (size_t col = 0; col < 2; ++col)
            EXPECT_FLOAT_EQ(R(row, col), 0.f);
}

TEST_F(MatMulOpForwardTest, ColumnVectorProduct) {
    // A = [[1,2,3],[4,5,6]]   v = [[1],[2],[3]]
    // Av = [[14],[32]]
    auto A = make_tensor({1,2,3,4,5,6}, {2,3});
    auto v = make_tensor({1,2,3},        {3,1});
    auto R = MatMulOp::forward(A, v);
    EXPECT_FLOAT_EQ(R(0,0), 14.f);
    EXPECT_FLOAT_EQ(R(1,0), 32.f);
}

TEST_F(MatMulOpForwardTest, InnerDimMismatchAsserts) {
    EXPECT_DEATH(MatMulOp::forward(Tensor::zeros(make_shape({2,3}), 2),
                                   Tensor::zeros(make_shape({4,2}), 2)), "");
}

TEST_F(MatMulOpForwardTest, NonTwoNdimAsserts) {
    EXPECT_DEATH(MatMulOp::forward(Tensor::zeros(make_shape({2,3,4}), 3),
                                   Tensor::zeros(make_shape({4,2,1}), 3)), "");
}

// ════════════════════════════════════════════════════════════════════════════
// 2. MatMulTransposedInput — non-contiguous inputs via T()
//
// forward() is stride-aware so T() inputs (non-contiguous) must produce the
// same result as explicitly transposed data would.
// ════════════════════════════════════════════════════════════════════════════

class MatMulTransposedInput : public ::testing::Test {};

TEST_F(MatMulTransposedInput, TransposedFirstInput) {
    // A = [[1,3],[2,4]]  →  A.T() = [[1,2],[3,4]]
    // A.T() @ I = A.T()
    auto A = make_tensor({1,3,2,4}, {2,2});
    auto I = make_tensor({1,0,0,1}, {2,2});
    auto AT = A.T();

    ASSERT_FALSE(AT.is_contiguous());

    auto R = MatMulOp::forward(AT, I);
    EXPECT_FLOAT_EQ(R(0,0), 1.f);
    EXPECT_FLOAT_EQ(R(0,1), 2.f);
    EXPECT_FLOAT_EQ(R(1,0), 3.f);
    EXPECT_FLOAT_EQ(R(1,1), 4.f);
}

TEST_F(MatMulTransposedInput, TransposedSecondInput) {
    // B = [[1,3],[2,4]]  →  B.T() = [[1,2],[3,4]]
    // I @ B.T() = B.T()
    auto I = make_tensor({1,0,0,1}, {2,2});
    auto B = make_tensor({1,3,2,4}, {2,2});
    auto BT = B.T();

    ASSERT_FALSE(BT.is_contiguous());

    auto R = MatMulOp::forward(I, BT);
    EXPECT_FLOAT_EQ(R(0,0), 1.f);
    EXPECT_FLOAT_EQ(R(0,1), 2.f);
    EXPECT_FLOAT_EQ(R(1,0), 3.f);
    EXPECT_FLOAT_EQ(R(1,1), 4.f);
}

// Both inputs transposed: A.T() @ B.T()
// This tests the interaction of two non-contiguous stride patterns simultaneously.
TEST_F(MatMulTransposedInput, BothInputsTransposed) {
    // A = [[1,2],[3,4]]  A.T() = [[1,3],[2,4]]
    // B = [[5,6],[7,8]]  B.T() = [[5,7],[6,8]]
    // A.T() @ B.T() = [[1*5+3*6, 1*7+3*8], [2*5+4*6, 2*7+4*8]]
    //               = [[23, 31], [34, 46]]
    auto A = make_tensor({1,2,3,4}, {2,2});
    auto B = make_tensor({5,6,7,8}, {2,2});

    ASSERT_FALSE(A.T().is_contiguous());
    ASSERT_FALSE(B.T().is_contiguous());

    auto R = MatMulOp::forward(A.T(), B.T());
    EXPECT_FLOAT_EQ(R(0,0), 23.f);
    EXPECT_FLOAT_EQ(R(0,1), 31.f);
    EXPECT_FLOAT_EQ(R(1,0), 34.f);
    EXPECT_FLOAT_EQ(R(1,1), 46.f);
}

TEST_F(MatMulTransposedInput, ContiguousAndTransposedAgree) {
    // Logical matrix X = [[1,2,3],[4,5,6]].
    // Stored column-major as X_col = [[1,4],[2,5],[3,6]] — then X_col.T() gives X.
    auto X_col = make_tensor({1,4,2,5,3,6}, {3,2});
    auto X_T   = X_col.T();                          // non-contiguous [2,3]
    auto B     = make_tensor({7,8,9,10,11,12}, {3,2});

    auto contiguous = make_tensor({1,2,3,4,5,6}, {2,3});

    auto slow_result = MatMulOp::forward(X_T, B);
    auto fast_result = MatMulOp::forward(contiguous, B);

    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_FLOAT_EQ(slow_result(i,j), fast_result(i,j))
                << "mismatch at [" << i << "," << j << "]";
}

// ════════════════════════════════════════════════════════════════════════════
// 3. MatMulOpBackwardTest — gradient values in isolation
//
// For C = A @ B with upstream gradient G (all ones):
//   dA = G  @ B^T
//   dB = A^T @ G
//
// A = [[1,2,3],[4,5,6]]   B = [[1],[2],[3]]   G = [[1],[1]]
//
// dA = [[1],[1]] @ [[1,2,3]]  = [[1,2,3],[1,2,3]]
// dB = [[1,4],[2,5],[3,6]] @ [[1],[1]] = [[5],[7],[9]]
// ════════════════════════════════════════════════════════════════════════════

class MatMulOpBackwardTest : public ::testing::Test {};

TEST_F(MatMulOpBackwardTest, ReturnsTwoGradients) {
    auto A    = make_tensor({1,2,3,4,5,6}, {2,3});
    auto B    = make_tensor({1,2,3},        {3,1});
    auto grad = make_tensor({1,1},          {2,1});

    auto grads = MatMulOp::backward(grad, A, B);
    EXPECT_EQ(grads.size(), 2u);
}

TEST_F(MatMulOpBackwardTest, GradAShape) {
    auto A    = make_tensor({1,2,3,4,5,6}, {2,3});
    auto B    = make_tensor({1,2,3},        {3,1});
    auto grad = make_tensor({1,1},          {2,1});

    auto grads = MatMulOp::backward(grad, A, B);
    EXPECT_EQ(grads[0].shape_at(0), 2u);
    EXPECT_EQ(grads[0].shape_at(1), 3u);
}

TEST_F(MatMulOpBackwardTest, GradBShape) {
    auto A    = make_tensor({1,2,3,4,5,6}, {2,3});
    auto B    = make_tensor({1,2,3},        {3,1});
    auto grad = make_tensor({1,1},          {2,1});

    auto grads = MatMulOp::backward(grad, A, B);
    EXPECT_EQ(grads[1].shape_at(0), 3u);
    EXPECT_EQ(grads[1].shape_at(1), 1u);
}

TEST_F(MatMulOpBackwardTest, GradAValues) {
    auto A    = make_tensor({1,2,3,4,5,6}, {2,3});
    auto B    = make_tensor({1,2,3},        {3,1});
    auto grad = make_tensor({1,1},          {2,1});

    auto grads = MatMulOp::backward(grad, A, B);

    // dA = [[1],[1]] @ [[1,2,3]] = [[1,2,3],[1,2,3]]
    EXPECT_FLOAT_EQ(grads[0](0,0), 1.f);
    EXPECT_FLOAT_EQ(grads[0](0,1), 2.f);
    EXPECT_FLOAT_EQ(grads[0](0,2), 3.f);
    EXPECT_FLOAT_EQ(grads[0](1,0), 1.f);
    EXPECT_FLOAT_EQ(grads[0](1,1), 2.f);
    EXPECT_FLOAT_EQ(grads[0](1,2), 3.f);
}

TEST_F(MatMulOpBackwardTest, GradBValues) {
    auto A    = make_tensor({1,2,3,4,5,6}, {2,3});
    auto B    = make_tensor({1,2,3},        {3,1});
    auto grad = make_tensor({1,1},          {2,1});

    auto grads = MatMulOp::backward(grad, A, B);

    // dB = [[1,4],[2,5],[3,6]] @ [[1],[1]] = [[5],[7],[9]]
    EXPECT_FLOAT_EQ(grads[1](0,0), 5.f);
    EXPECT_FLOAT_EQ(grads[1](1,0), 7.f);
    EXPECT_FLOAT_EQ(grads[1](2,0), 9.f);
}

TEST_F(MatMulOpBackwardTest, GradAValues2x2) {
    // A = [[1,2],[3,4]]   B = [[5,6],[7,8]]   G = [[1,1],[1,1]]
    // dA = G @ B^T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
    auto A    = make_tensor({1,2,3,4}, {2,2});
    auto B    = make_tensor({5,6,7,8}, {2,2});
    auto grad = Tensor::ones(make_shape({2,2}), 2);

    auto grads = MatMulOp::backward(grad, A, B);

    EXPECT_FLOAT_EQ(grads[0](0,0), 11.f);
    EXPECT_FLOAT_EQ(grads[0](0,1), 15.f);
    EXPECT_FLOAT_EQ(grads[0](1,0), 11.f);
    EXPECT_FLOAT_EQ(grads[0](1,1), 15.f);
}

TEST_F(MatMulOpBackwardTest, GradBValues2x2) {
    // dB = A^T @ G = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
    auto A    = make_tensor({1,2,3,4}, {2,2});
    auto B    = make_tensor({5,6,7,8}, {2,2});
    auto grad = Tensor::ones(make_shape({2,2}), 2);

    auto grads = MatMulOp::backward(grad, A, B);

    EXPECT_FLOAT_EQ(grads[1](0,0), 4.f);
    EXPECT_FLOAT_EQ(grads[1](0,1), 4.f);
    EXPECT_FLOAT_EQ(grads[1](1,0), 6.f);
    EXPECT_FLOAT_EQ(grads[1](1,1), 6.f);
}

// ════════════════════════════════════════════════════════════════════════════
// 4. MatMulFuncTest — matmul() autograd wiring
// ════════════════════════════════════════════════════════════════════════════

class MatMulFuncTest : public ::testing::Test {};

TEST_F(MatMulFuncTest, ProducesCorrectValues) {
    auto A   = make_tensor({1,2,3,4,5,6}, {2,3});
    auto B   = make_tensor({1,2,3},        {3,1});
    auto C   = matmul(A, B);
    // [[1*1+2*2+3*3],[4*1+5*2+6*3]] = [[14],[32]]
    EXPECT_FLOAT_EQ(C(0,0), 14.f);
    EXPECT_FLOAT_EQ(C(1,0), 32.f);
}

TEST_F(MatMulFuncTest, NoRequiresGradProducesNoMeta) {
    auto C = matmul(make_tensor({1,2,3,4}, {2,2}),
                    make_tensor({1,2,3,4}, {2,2}));
    EXPECT_EQ(C.autograd_meta, nullptr);
    EXPECT_FALSE(C.requires_grad());
}

TEST_F(MatMulFuncTest, BothRequireGrad) {
    auto C = matmul(make_tensor({1,2,3,4}, {2,2}, true),
                    make_tensor({1,2,3,4}, {2,2}, true));
    EXPECT_TRUE(C.requires_grad());
}

TEST_F(MatMulFuncTest, OnlyARequiresGrad) {
    auto C = matmul(make_tensor({1,2,3,4}, {2,2}, true),
                    make_tensor({1,2,3,4}, {2,2}, false));
    EXPECT_TRUE(C.requires_grad());
}

TEST_F(MatMulFuncTest, OnlyBRequiresGrad) {
    auto C = matmul(make_tensor({1,2,3,4}, {2,2}, false),
                    make_tensor({1,2,3,4}, {2,2}, true));
    EXPECT_TRUE(C.requires_grad());
}

TEST_F(MatMulFuncTest, NoGradBeforeBackward) {
    auto C = matmul(make_tensor({1,2,3,4}, {2,2}, true),
                    make_tensor({1,2,3,4}, {2,2}, true));
    EXPECT_FALSE(C.has_grad());
}

// ════════════════════════════════════════════════════════════════════════════
// 5. MatMulAutogradTest — full backward() through matmul()
// ════════════════════════════════════════════════════════════════════════════

class MatMulAutogradTest : public ::testing::Test {};

TEST_F(MatMulAutogradTest, GradOnAAfterBackward) {
    auto A = make_tensor({1,2,3,4,5,6}, {2,3}, true);
    auto B = make_tensor({1,2,3},        {3,1}, true);
    auto C = matmul(A, B);
    backward(C);

    ASSERT_TRUE(A.has_grad());
    // dA = [[1],[1]] @ [[1,2,3]] = [[1,2,3],[1,2,3]]
    EXPECT_FLOAT_EQ(A.grad()(0,0), 1.f);
    EXPECT_FLOAT_EQ(A.grad()(0,1), 2.f);
    EXPECT_FLOAT_EQ(A.grad()(0,2), 3.f);
    EXPECT_FLOAT_EQ(A.grad()(1,0), 1.f);
    EXPECT_FLOAT_EQ(A.grad()(1,1), 2.f);
    EXPECT_FLOAT_EQ(A.grad()(1,2), 3.f);
}

TEST_F(MatMulAutogradTest, GradOnBAfterBackward) {
    auto A = make_tensor({1,2,3,4,5,6}, {2,3}, true);
    auto B = make_tensor({1,2,3},        {3,1}, true);
    auto C = matmul(A, B);
    backward(C);

    ASSERT_TRUE(B.has_grad());
    // dB = A^T @ [[1],[1]] = [[5],[7],[9]]
    EXPECT_FLOAT_EQ(B.grad()(0,0), 5.f);
    EXPECT_FLOAT_EQ(B.grad()(1,0), 7.f);
    EXPECT_FLOAT_EQ(B.grad()(2,0), 9.f);
}

TEST_F(MatMulAutogradTest, OnlyAReceivesGrad) {
    auto A = make_tensor({1,2,3,4}, {2,2}, true);
    auto B = make_tensor({1,2,3,4}, {2,2}, false);
    auto C = matmul(A, B);
    backward(C);
    EXPECT_TRUE(A.has_grad());
    EXPECT_FALSE(B.has_grad());
}

TEST_F(MatMulAutogradTest, OnlyBReceivesGrad) {
    auto A = make_tensor({1,2,3,4}, {2,2}, false);
    auto B = make_tensor({1,2,3,4}, {2,2}, true);
    auto C = matmul(A, B);
    backward(C);
    EXPECT_FALSE(A.has_grad());
    EXPECT_TRUE(B.has_grad());
}

TEST_F(MatMulAutogradTest, GradShapeMatchesInput) {
    auto A = make_tensor({1,2,3,4,5,6}, {2,3}, true);
    auto B = make_tensor({1,2,3,4,5,6}, {3,2}, true);
    auto C = matmul(A, B);
    backward(C);
    EXPECT_EQ(A.grad().shape_at(0), 2u);  EXPECT_EQ(A.grad().shape_at(1), 3u);
    EXPECT_EQ(B.grad().shape_at(0), 3u);  EXPECT_EQ(B.grad().shape_at(1), 2u);
}

TEST_F(MatMulAutogradTest, ForwardValuesUnchangedAfterBackward) {
    auto A = make_tensor({1,2,3,4}, {2,2}, true);
    auto B = make_tensor({5,6,7,8}, {2,2}, true);
    auto C = matmul(A, B);
    float c00 = C(0,0), c01 = C(0,1), c10 = C(1,0), c11 = C(1,1);
    backward(C);
    EXPECT_FLOAT_EQ(C(0,0), c00);
    EXPECT_FLOAT_EQ(C(0,1), c01);
    EXPECT_FLOAT_EQ(C(1,0), c10);
    EXPECT_FLOAT_EQ(C(1,1), c11);
}

// ════════════════════════════════════════════════════════════════════════════
// 6. MatMulTilingTest — tiling boundary conditions
//
// The kernel tiles with T=64.  Shapes that cross a tile boundary (e.g. 65)
// exercise the min(m0+T, M) clamping in all three loop dimensions.
// ════════════════════════════════════════════════════════════════════════════

class MatMulTilingTest : public ::testing::Test {};

// A filled-with-ones [65×65] times itself gives [65×65] filled with 65.
// This covers the case where M, N, and K each require exactly two tile
// iterations (tile 0..63 and remainder 64..64), exercising the boundary clamp.
TEST_F(MatMulTilingTest, ShapeJustAboveTileBoundary) {
    constexpr size_t N = 65;
    auto A = Tensor::ones(make_shape({N, N}), 2);
    auto B = Tensor::ones(make_shape({N, N}), 2);
    auto C = MatMulOp::forward(A, B);

    EXPECT_EQ(C.shape_at(0), N);
    EXPECT_EQ(C.shape_at(1), N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            EXPECT_FLOAT_EQ(C(i, j), static_cast<float>(N))
                << "mismatch at [" << i << "," << j << "]";
}

// Extreme aspect ratio: tall-times-wide ([1×K] @ [K×1] = [1×1] scalar matmul).
// Exercises the case where M=1 and N=1 so no tiling occurs in those dims,
// but K may span many tiles.
TEST_F(MatMulTilingTest, ExtremeAspectRatioRowTimesColumn) {
    constexpr size_t K = 256;
    auto A = Tensor::ones(make_shape({1, K}), 2);
    auto B = Tensor::ones(make_shape({K, 1}), 2);
    auto C = MatMulOp::forward(A, B);

    EXPECT_EQ(C.shape_at(0), 1u);
    EXPECT_EQ(C.shape_at(1), 1u);
    EXPECT_FLOAT_EQ(C(0, 0), static_cast<float>(K));
}
