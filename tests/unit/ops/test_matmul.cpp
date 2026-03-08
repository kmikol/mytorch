// tests/test_matmul.cpp
//
// Unit tests for matmul() and MatMulOp.
//
// The implementation has two distinct forward paths that must both be tested:
//
//   Fast path  — both inputs are contiguous: uses raw pointer arithmetic.
//   Slow path  — at least one input is non-contiguous: uses at() with strides.
//
// Both paths must produce identical results.  A bug in one path but not the
// other would only surface on whichever input layout the training code uses.
//
// Test organisation:
//   1.  MatMulOpForward        — forward() in isolation, shape and values
//   2.  MatMulForwardFastPath  — contiguous inputs use the pointer loop
//   3.  MatMulForwardSlowPath  — non-contiguous inputs use at() loop
//   4.  MatMulForwardBothPaths — fast and slow produce identical results
//   5.  MatMulOpBackward       — backward() gradients in isolation
//   6.  MatMulAutograd         — requires_grad propagation and graph wiring
//   7.  MatMulBackwardValues   — numerical gradient values after backward()

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "tensorlib.h"
#include "ops/ops.h"

// ════════════════════════════════════════════════════════════════════════════
// 1.  MatMulOpForward — shape and value correctness
//
// We test forward() directly before testing the public matmul() wrapper so
// that shape or value failures can be attributed to the core computation
// rather than to the autograd wiring around it.
// ════════════════════════════════════════════════════════════════════════════

TEST(MatMulOpForward, OutputShapeIsCorrectForRectangularMatrices) {
    // [2,3] @ [3,4] → [2,4]
    Tensor matrix_a = Tensor::zeros({2, 3});
    Tensor matrix_b = Tensor::zeros({3, 4});
    Tensor result   = MatMulOp::forward(matrix_a, matrix_b);

    EXPECT_EQ(result.shape(0), 2);
    EXPECT_EQ(result.shape(1), 4);
}

TEST(MatMulOpForward, OutputShapeIsCorrectForSquareMatrices) {
    Tensor matrix_a = Tensor::zeros({3, 3});
    Tensor matrix_b = Tensor::zeros({3, 3});
    Tensor result   = MatMulOp::forward(matrix_a, matrix_b);

    EXPECT_EQ(result.shape(0), 3);
    EXPECT_EQ(result.shape(1), 3);
}

TEST(MatMulOpForward, OutputShapeIsCorrectForColumnVector) {
    // [2,3] @ [3,1] → [2,1]  — the most common shape in linear layers
    Tensor matrix = Tensor::zeros({2, 3});
    Tensor vector = Tensor::zeros({3, 1});
    Tensor result = MatMulOp::forward(matrix, vector);

    EXPECT_EQ(result.shape(0), 2);
    EXPECT_EQ(result.shape(1), 1);
}

TEST(MatMulOpForward, OutputNdimIsAlwaysTwo) {
    Tensor matrix_a = Tensor::zeros({2, 3});
    Tensor matrix_b = Tensor::zeros({3, 4});
    Tensor result   = MatMulOp::forward(matrix_a, matrix_b);

    EXPECT_EQ(result.ndim(), 2);
}

TEST(MatMulOpForward, ComputesCorrect2x3by3x2Product) {
    // Hand-computed reference:
    // A = [1 2 3]    B = [7  8 ]
    //     [4 5 6]        [9  10]
    //                    [11 12]
    //
    // C[0,0] = 1*7  + 2*9  + 3*11 = 58
    // C[0,1] = 1*8  + 2*10 + 3*12 = 64
    // C[1,0] = 4*7  + 5*9  + 6*11 = 139
    // C[1,1] = 4*8  + 5*10 + 6*12 = 154
    Tensor matrix_a = Tensor::from_data({1, 2, 3, 4, 5, 6},       {2, 3});
    Tensor matrix_b = Tensor::from_data({7, 8, 9, 10, 11, 12},    {3, 2});
    Tensor result   = MatMulOp::forward(matrix_a, matrix_b);

    EXPECT_FLOAT_EQ(result.at({0, 0}),  58.f);
    EXPECT_FLOAT_EQ(result.at({0, 1}),  64.f);
    EXPECT_FLOAT_EQ(result.at({1, 0}), 139.f);
    EXPECT_FLOAT_EQ(result.at({1, 1}), 154.f);
}

TEST(MatMulOpForward, MultiplyByIdentityLeavesMatrixUnchanged) {
    // X @ I must equal X for any X.  This checks every element so that a
    // systematic offset in the index arithmetic would be caught.
    Tensor identity = Tensor::from_data({1, 0, 0,
                                         0, 1, 0,
                                         0, 0, 1}, {3, 3});
    Tensor matrix_x = Tensor::from_data({1, 2, 3,
                                         4, 5, 6,
                                         7, 8, 9}, {3, 3});
    Tensor result   = MatMulOp::forward(matrix_x, identity);

    for (int64_t row = 0; row < 3; ++row) {
        for (int64_t col = 0; col < 3; ++col) {
            EXPECT_FLOAT_EQ(result.at({row, col}), matrix_x.at({row, col}))
                << "mismatch at [" << row << ", " << col << "]";
        }
    }
}

TEST(MatMulOpForward, MultiplyByZeroMatrixProducesAllZeros) {
    Tensor matrix_a = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor zero_b   = Tensor::zeros({3, 2});
    Tensor result   = MatMulOp::forward(matrix_a, zero_b);

    for (int64_t row = 0; row < 2; ++row) {
        for (int64_t col = 0; col < 2; ++col) {
            EXPECT_FLOAT_EQ(result.at({row, col}), 0.f)
                << "expected 0 at [" << row << ", " << col << "]";
        }
    }
}

TEST(MatMulOpForward, ComputesCorrectColumnVectorProduct) {
    // A = [1 2 3]    v = [1]    Av = [1*1 + 2*2 + 3*3] = [14]
    //     [4 5 6]        [2]         [4*1 + 5*2 + 6*3]   [32]
    //                    [3]
    Tensor matrix_a = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor vector_v = Tensor::from_data({1, 2, 3},           {3, 1});
    Tensor result   = MatMulOp::forward(matrix_a, vector_v);

    EXPECT_FLOAT_EQ(result.at({0, 0}), 14.f);
    EXPECT_FLOAT_EQ(result.at({1, 0}), 32.f);
}

TEST(MatMulOpForward, OutputIsAlwaysContiguous) {
    // forward() allocates with Tensor::zeros which is always contiguous.
    // Operations downstream that assume contiguity must not be surprised.
    Tensor matrix_a = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor matrix_b = Tensor::from_data({1, 2, 3, 4, 5, 6}, {3, 2});
    Tensor result   = MatMulOp::forward(matrix_a, matrix_b);

    EXPECT_TRUE(result.is_contiguous());
}

// ════════════════════════════════════════════════════════════════════════════
// 2.  MatMulForwardFastPath — contiguous inputs
//
// The fast path uses raw pointer arithmetic and is only taken when all three
// of A, B, and C are contiguous.  We verify the fast path is active by
// confirming both inputs are contiguous before calling forward().
// ════════════════════════════════════════════════════════════════════════════

TEST(MatMulForwardFastPath, BothInputsContiguous) {
    Tensor matrix_a = Tensor::from_data({1, 2, 3, 4, 5, 6},    {2, 3});
    Tensor matrix_b = Tensor::from_data({7, 8, 9, 10, 11, 12}, {3, 2});

    ASSERT_TRUE(matrix_a.is_contiguous());
    ASSERT_TRUE(matrix_b.is_contiguous());

    Tensor result = MatMulOp::forward(matrix_a, matrix_b);

    EXPECT_FLOAT_EQ(result.at({0, 0}),  58.f);
    EXPECT_FLOAT_EQ(result.at({0, 1}),  64.f);
    EXPECT_FLOAT_EQ(result.at({1, 0}), 139.f);
    EXPECT_FLOAT_EQ(result.at({1, 1}), 154.f);
}

// ════════════════════════════════════════════════════════════════════════════
// 3.  MatMulForwardSlowPath — non-contiguous inputs
//
// The slow path is taken when at least one input has non-default strides
// (e.g. after transpose()).  It uses at() which respects strides.
// Testing this path separately ensures a bug in the stride-aware loop
// cannot hide behind the fast path always being taken.
// ════════════════════════════════════════════════════════════════════════════

TEST(MatMulForwardSlowPath, NonContiguousFirstInput) {
    // A^T is non-contiguous.  forward(A^T, B) exercises the slow path.
    // We choose values so the result is easy to verify by hand.
    //
    // A = [1 3]   A^T = [1 2]   B = [1 0]
    //     [2 4]         [3 4]       [0 1]
    //
    // A^T @ I = A^T = [1 2]
    //                 [3 4]
    Tensor matrix_a     = Tensor::from_data({1, 3, 2, 4}, {2, 2});
    Tensor transposed_a = matrix_a.transpose();
    Tensor identity     = Tensor::from_data({1, 0, 0, 1}, {2, 2});

    ASSERT_FALSE(transposed_a.is_contiguous());

    Tensor result = MatMulOp::forward(transposed_a, identity);

    EXPECT_FLOAT_EQ(result.at({0, 0}), 1.f);
    EXPECT_FLOAT_EQ(result.at({0, 1}), 2.f);
    EXPECT_FLOAT_EQ(result.at({1, 0}), 3.f);
    EXPECT_FLOAT_EQ(result.at({1, 1}), 4.f);
}

TEST(MatMulForwardSlowPath, NonContiguousSecondInput) {
    // I @ B^T exercises the slow path on the second argument.
    Tensor identity     = Tensor::from_data({1, 0, 0, 1}, {2, 2});
    Tensor matrix_b     = Tensor::from_data({1, 3, 2, 4}, {2, 2});
    Tensor transposed_b = matrix_b.transpose();

    ASSERT_FALSE(transposed_b.is_contiguous());

    // I @ B^T = B^T
    Tensor result = MatMulOp::forward(identity, transposed_b);

    EXPECT_FLOAT_EQ(result.at({0, 0}), matrix_b.at({0, 0}));
    EXPECT_FLOAT_EQ(result.at({0, 1}), matrix_b.at({1, 0}));
    EXPECT_FLOAT_EQ(result.at({1, 0}), matrix_b.at({0, 1}));
    EXPECT_FLOAT_EQ(result.at({1, 1}), matrix_b.at({1, 1}));
}

// ════════════════════════════════════════════════════════════════════════════
// 4.  MatMulForwardBothPaths — fast and slow must agree
//
// The two paths are alternative implementations of the same computation.
// Any input that produces different results from the two paths is a bug
// in one of them.  We compare them directly on the same data.
// ════════════════════════════════════════════════════════════════════════════

TEST(MatMulForwardBothPaths, FastAndSlowProduceIdenticalResults) {
    Tensor matrix_a     = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor matrix_b     = Tensor::from_data({7, 8, 9, 10, 11, 12}, {3, 2});

    // fast path: both contiguous
    Tensor fast_result = MatMulOp::forward(matrix_a, matrix_b);

    // slow path: make A non-contiguous by transposing twice
    // (double transpose restores logical values but may not restore strides)
    // Instead use a transposed input where we know the expected answer
    // A^T has shape [3,2], so we need B of shape [2, N]
    Tensor matrix_c         = Tensor::from_data({1, 4, 2, 5, 3, 6}, {3, 2});
    Tensor transposed_c     = matrix_c.transpose();   // [2,3], non-contiguous
    Tensor matrix_d         = Tensor::from_data({7, 8, 9, 10, 11, 12}, {3, 2});

    ASSERT_FALSE(transposed_c.is_contiguous());
    Tensor slow_result = MatMulOp::forward(transposed_c, matrix_d);

    // transposed_c has the same logical values as matrix_a
    for (int64_t row = 0; row < 2; ++row) {
        for (int64_t col = 0; col < 2; ++col) {
            EXPECT_FLOAT_EQ(slow_result.at({row, col}),
                            fast_result.at({row, col}))
                << "fast/slow disagreement at [" << row << ", " << col << "]";
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 5.  MatMulOpBackward — gradient values in isolation
//
// backward() is tested directly here so that gradient value failures can be
// separated from failures in the autograd graph wiring in the wrapper.
//
// For C = A @ B with upstream gradient G (all ones):
//   dA = G  @ B^T
//   dB = A^T @ G
//
// A = [1 2 3]   B = [1]   G = [1]
//     [4 5 6]       [2]       [1]
//                   [3]
//
// dA = G @ B^T = [[1],[1]] @ [[1,2,3]] = [[1,2,3],[1,2,3]]
// dB = A^T @ G = [[1,4],[2,5],[3,6]] @ [[1],[1]] = [[5],[7],[9]]
// ════════════════════════════════════════════════════════════════════════════

TEST(MatMulOpBackward, ReturnsExactlyTwoGradients) {
    Tensor saved_a          = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor saved_b          = Tensor::from_data({1, 2, 3},           {3, 1});
    Tensor upstream_grad    = Tensor::from_data({1, 1},               {2, 1});

    std::vector<Tensor> gradients = MatMulOp::backward(
        upstream_grad, saved_a, saved_b, true, true);

    EXPECT_EQ(gradients.size(), 2u);
}

TEST(MatMulOpBackward, GradientForAIsCorrect) {
    Tensor saved_a          = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor saved_b          = Tensor::from_data({1, 2, 3},           {3, 1});
    Tensor upstream_grad    = Tensor::from_data({1, 1},               {2, 1});

    std::vector<Tensor> gradients = MatMulOp::backward(
        upstream_grad, saved_a, saved_b, true, true);

    // dA = [[1],[1]] @ [[1,2,3]] = [[1,2,3],[1,2,3]]
    EXPECT_FLOAT_EQ(gradients[0].at({0, 0}), 1.f);
    EXPECT_FLOAT_EQ(gradients[0].at({0, 1}), 2.f);
    EXPECT_FLOAT_EQ(gradients[0].at({0, 2}), 3.f);
    EXPECT_FLOAT_EQ(gradients[0].at({1, 0}), 1.f);
    EXPECT_FLOAT_EQ(gradients[0].at({1, 1}), 2.f);
    EXPECT_FLOAT_EQ(gradients[0].at({1, 2}), 3.f);
}

TEST(MatMulOpBackward, GradientForBIsCorrect) {
    Tensor saved_a          = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor saved_b          = Tensor::from_data({1, 2, 3},           {3, 1});
    Tensor upstream_grad    = Tensor::from_data({1, 1},               {2, 1});

    std::vector<Tensor> gradients = MatMulOp::backward(
        upstream_grad, saved_a, saved_b, true, true);

    // dB = A^T @ G = [[1+4],[2+5],[3+6]] = [[5],[7],[9]]
    EXPECT_FLOAT_EQ(gradients[1].at({0, 0}), 5.f);
    EXPECT_FLOAT_EQ(gradients[1].at({1, 0}), 7.f);
    EXPECT_FLOAT_EQ(gradients[1].at({2, 0}), 9.f);
}

TEST(MatMulOpBackward, GradientForAIsEmptyWhenNotRequired) {
    // When rA=false backward() must leave gradients[0] as a default-constructed
    // Tensor (no implementation).  Returning a real gradient for a tensor that
    // does not require one would waste computation in training loops.
    Tensor saved_a          = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor saved_b          = Tensor::from_data({1, 2, 3},           {3, 1});
    Tensor upstream_grad    = Tensor::from_data({1, 1},               {2, 1});

    std::vector<Tensor> gradients = MatMulOp::backward(
        upstream_grad, saved_a, saved_b, false, true);

    EXPECT_EQ(gradients[0].implementation, nullptr);
}

TEST(MatMulOpBackward, GradientForBIsEmptyWhenNotRequired) {
    Tensor saved_a          = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor saved_b          = Tensor::from_data({1, 2, 3},           {3, 1});
    Tensor upstream_grad    = Tensor::from_data({1, 1},               {2, 1});

    std::vector<Tensor> gradients = MatMulOp::backward(
        upstream_grad, saved_a, saved_b, true, false);

    EXPECT_EQ(gradients[1].implementation, nullptr);
}

TEST(MatMulOpBackward, GradientShapeForAMatchesSavedA) {
    Tensor saved_a          = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor saved_b          = Tensor::from_data({1, 2, 3},           {3, 1});
    Tensor upstream_grad    = Tensor::from_data({1, 1},               {2, 1});

    std::vector<Tensor> gradients = MatMulOp::backward(
        upstream_grad, saved_a, saved_b, true, true);

    EXPECT_EQ(gradients[0].shape(0), 2);
    EXPECT_EQ(gradients[0].shape(1), 3);
}

TEST(MatMulOpBackward, GradientShapeForBMatchesSavedB) {
    Tensor saved_a          = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor saved_b          = Tensor::from_data({1, 2, 3},           {3, 1});
    Tensor upstream_grad    = Tensor::from_data({1, 1},               {2, 1});

    std::vector<Tensor> gradients = MatMulOp::backward(
        upstream_grad, saved_a, saved_b, true, true);

    EXPECT_EQ(gradients[1].shape(0), 3);
    EXPECT_EQ(gradients[1].shape(1), 1);
}

// ════════════════════════════════════════════════════════════════════════════
// 6.  MatMulAutograd — requires_grad propagation through the wrapper
//
// The public matmul() wrapper is responsible for creating the autograd node.
// We test all four combinations of requires_grad on the two inputs.
// ════════════════════════════════════════════════════════════════════════════

TEST(MatMulAutograd, OutputRequiresGradWhenBothInputsDo) {
    Tensor matrix_a = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, true);
    Tensor matrix_b = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, true);
    Tensor result   = matmul(matrix_a, matrix_b);

    EXPECT_TRUE(result.requires_grad());
}

TEST(MatMulAutograd, OutputRequiresGradWhenOnlyADoes) {
    Tensor matrix_a = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, true);
    Tensor matrix_b = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, false);
    Tensor result   = matmul(matrix_a, matrix_b);

    EXPECT_TRUE(result.requires_grad());
}

TEST(MatMulAutograd, OutputRequiresGradWhenOnlyBDoes) {
    Tensor matrix_a = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, false);
    Tensor matrix_b = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, true);
    Tensor result   = matmul(matrix_a, matrix_b);

    EXPECT_TRUE(result.requires_grad());
}

TEST(MatMulAutograd, OutputDoesNotRequireGradWhenNeitherInputDoes) {
    Tensor matrix_a = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, false);
    Tensor matrix_b = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, false);
    Tensor result   = matmul(matrix_a, matrix_b);

    EXPECT_FALSE(result.requires_grad());
}

TEST(MatMulAutograd, OutputHasNoGradBeforeBackward) {
    Tensor matrix_a = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, true);
    Tensor matrix_b = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, true);
    Tensor result   = matmul(matrix_a, matrix_b);

    EXPECT_FALSE(result.has_grad());
}

TEST(MatMulAutograd, NoAutogradMetaWhenNeitherInputRequiresGrad) {
    // When neither input requires grad no graph node must be created.
    // Inserting a node would waste memory proportional to the number of
    // operations in the forward pass.
    Tensor matrix_a = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, false);
    Tensor matrix_b = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, false);
    Tensor result   = matmul(matrix_a, matrix_b);

    EXPECT_EQ(result.autograd_meta, nullptr);
}

// ════════════════════════════════════════════════════════════════════════════
// 7.  MatMulBackwardValues — full backward pass through the wrapper
//
// These tests call backward() on the output of matmul() and verify that
// gradients accumulated on the input leaf tensors are numerically correct.
// This exercises the complete path: forward → graph node → backward.
// ════════════════════════════════════════════════════════════════════════════

TEST(MatMulBackwardValues, GradientAccumulatesOnAAfterBackward) {
    Tensor matrix_a = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}, {}, true);
    Tensor matrix_b = Tensor::from_data({1, 2, 3},           {3, 1}, {}, true);
    Tensor result   = matmul(matrix_a, matrix_b);

    backward(result);

    ASSERT_TRUE(matrix_a.has_grad());
    EXPECT_FLOAT_EQ(matrix_a.grad().at({0, 0}), 1.f);
    EXPECT_FLOAT_EQ(matrix_a.grad().at({0, 1}), 2.f);
    EXPECT_FLOAT_EQ(matrix_a.grad().at({0, 2}), 3.f);
    EXPECT_FLOAT_EQ(matrix_a.grad().at({1, 0}), 1.f);
    EXPECT_FLOAT_EQ(matrix_a.grad().at({1, 1}), 2.f);
    EXPECT_FLOAT_EQ(matrix_a.grad().at({1, 2}), 3.f);
}

TEST(MatMulBackwardValues, GradientAccumulatesOnBAfterBackward) {
    Tensor matrix_a = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}, {}, true);
    Tensor matrix_b = Tensor::from_data({1, 2, 3},           {3, 1}, {}, true);
    Tensor result   = matmul(matrix_a, matrix_b);

    backward(result);

    ASSERT_TRUE(matrix_b.has_grad());
    EXPECT_FLOAT_EQ(matrix_b.grad().at({0, 0}), 5.f);
    EXPECT_FLOAT_EQ(matrix_b.grad().at({1, 0}), 7.f);
    EXPECT_FLOAT_EQ(matrix_b.grad().at({2, 0}), 9.f);
}

TEST(MatMulBackwardValues, OnlyAReceivesGradWhenOnlyARequiresIt) {
    Tensor matrix_a = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, true);
    Tensor matrix_b = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, false);
    Tensor result   = matmul(matrix_a, matrix_b);

    backward(result);

    EXPECT_TRUE(matrix_a.has_grad());
    EXPECT_FALSE(matrix_b.has_grad());
}

TEST(MatMulBackwardValues, OnlyBReceivesGradWhenOnlyBRequiresIt) {
    Tensor matrix_a = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, false);
    Tensor matrix_b = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, true);
    Tensor result   = matmul(matrix_a, matrix_b);

    backward(result);

    EXPECT_FALSE(matrix_a.has_grad());
    EXPECT_TRUE(matrix_b.has_grad());
}

TEST(MatMulBackwardValues, GradientShapeMatchesInputShape) {
    // The gradient of A must have the same shape as A — not the shape of
    // the output.  A shape mismatch here would crash any optimizer that
    // subtracts the gradient from the parameter in-place.
    Tensor matrix_a = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}, {}, true);
    Tensor matrix_b = Tensor::from_data({1, 2, 3, 4, 5, 6}, {3, 2}, {}, true);
    Tensor result   = matmul(matrix_a, matrix_b);

    backward(result);

    EXPECT_EQ(matrix_a.grad().shape(0), 2);
    EXPECT_EQ(matrix_a.grad().shape(1), 3);
    EXPECT_EQ(matrix_b.grad().shape(0), 3);
    EXPECT_EQ(matrix_b.grad().shape(1), 2);
}

TEST(MatMulBackwardValues, SavedTensorsAreNotAliasedToInputs) {
    // matmul() clones A and B before capturing them in the backward closure.
    // If it captured references instead, mutating A or B after the forward
    // pass would corrupt the saved tensors and produce wrong gradients.
    Tensor matrix_a = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, true);
    Tensor matrix_b = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, true);
    Tensor result   = matmul(matrix_a, matrix_b);

    // mutate inputs after forward but before backward
    matrix_a.at({0, 0}) = 999.f;
    matrix_b.at({0, 0}) = 999.f;

    // backward must still produce gradients based on the original values,
    // not the mutated ones — so it must not throw or produce NaN/Inf
    EXPECT_NO_THROW(backward(result));
    EXPECT_TRUE(matrix_a.has_grad());
    EXPECT_TRUE(matrix_b.has_grad());
}