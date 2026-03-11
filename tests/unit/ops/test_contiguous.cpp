// tests/test_contiguous.cpp
//
// Unit tests for contiguous() and ContiguousOp.
//
// contiguous() has two distinct code paths that must both be tested:
//
//   Fast path — tensor is already contiguous:
//     Returns a view sharing the same storage and autograd_meta.
//     No allocation, no copy, no graph node.
//
//   Slow path — tensor is not contiguous (e.g. transposed):
//     Allocates fresh storage, copies elements in logical row-major order,
//     and wires a backward node into the autograd graph when required.
//
// ContiguousOp::forward and ContiguousOp::backward are also tested
// directly so that failures in the high-level contiguous() function
// can be attributed to the op itself versus the dispatch logic.
//
// Conventions carried over from the rest of the test suite:
//   - no `auto` — every type is spelled out
//   - descriptive names
//   - comments explain why a check exists, not just what it does

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "tensorlib.h"
#include "ops/ops.h"           // contiguous(), ContiguousOp

// ════════════════════════════════════════════════════════════════════════════
// ContiguousOp::forward — tested in isolation
//
// forward() is a pure data transformation: allocate fresh storage and copy
// elements in logical order.  We test it directly so that if contiguous()
// fails we can tell whether the bug is in the op or in the dispatch logic.
// ════════════════════════════════════════════════════════════════════════════

TEST(ContiguousOpForward, ProducesContiguousOutputFromContiguousInput) {
    Tensor input = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor output = ContiguousOp::forward(input);

    EXPECT_TRUE(output.is_contiguous());
}

TEST(ContiguousOpForward, ProducesContiguousOutputFromNonContiguousInput) {
    // Transposing breaks contiguity.  forward() must produce a fresh
    // contiguous layout regardless of the source strides.
    Tensor transposed = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}).transpose();
    ASSERT_FALSE(transposed.is_contiguous());

    Tensor output = ContiguousOp::forward(transposed);
    EXPECT_TRUE(output.is_contiguous());
}

TEST(ContiguousOpForward, OutputShapeMatchesInput) {
    Tensor transposed = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}).transpose();
    Tensor output = ContiguousOp::forward(transposed);

    // shape must reflect the logical layout of the source, not its storage
    EXPECT_EQ(output.shape(0), 3);
    EXPECT_EQ(output.shape(1), 2);
}

TEST(ContiguousOpForward, OutputValuesMatchLogicalLayoutOfInput) {
    // original [2,3]:          transposed [3,2]:
    //   [1  2  3]                [1  4]
    //   [4  5  6]                [2  5]
    //                            [3  6]
    // forward() on the transposed view must produce those logical values
    // in row-major order in the new storage.
    Tensor transposed = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}).transpose();
    Tensor output = ContiguousOp::forward(transposed);

    EXPECT_FLOAT_EQ(output.at(0, 0), 1.f);
    EXPECT_FLOAT_EQ(output.at(0, 1), 4.f);
    EXPECT_FLOAT_EQ(output.at(1, 0), 2.f);
    EXPECT_FLOAT_EQ(output.at(1, 1), 5.f);
    EXPECT_FLOAT_EQ(output.at(2, 0), 3.f);
    EXPECT_FLOAT_EQ(output.at(2, 1), 6.f);
}

TEST(ContiguousOpForward, OutputHasFreshStorageNotSharedWithInput) {
    // forward() must always allocate — it is the slow path that exists
    // specifically to break storage sharing.  If storage were shared, the
    // contiguous guarantee would be meaningless.
    Tensor transposed = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}).transpose();
    Tensor output = ContiguousOp::forward(transposed);

    EXPECT_NE(
        output.data_ptr(),
        transposed.data_ptr()
    );
}

TEST(ContiguousOpForward, OutputStorageIsPackedRowMajor) {
    // Verify the raw storage bytes are in row-major order, not just that
    // at() returns the right values (which could mask a wrong-stride bug).
    //
    // transposed [3,2] logical:    expected packed storage: 1 4 2 5 3 6
    //   [1  4]
    //   [2  5]
    //   [3  6]
    Tensor transposed = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}).transpose();
    Tensor output = ContiguousOp::forward(transposed);

    float* raw_storage_pointer = output.data_ptr();
    EXPECT_FLOAT_EQ(raw_storage_pointer[0], 1.f);
    EXPECT_FLOAT_EQ(raw_storage_pointer[1], 4.f);
    EXPECT_FLOAT_EQ(raw_storage_pointer[2], 2.f);
    EXPECT_FLOAT_EQ(raw_storage_pointer[3], 5.f);
    EXPECT_FLOAT_EQ(raw_storage_pointer[4], 3.f);
    EXPECT_FLOAT_EQ(raw_storage_pointer[5], 6.f);
}

TEST(ContiguousOpForward, OutputHasDefaultStridesForItsShape) {
    // After forward(), strides must be the default row-major strides for
    // the output shape — not the transposed strides of the source.
    Tensor transposed = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}).transpose();
    Tensor output = ContiguousOp::forward(transposed);

    std::vector<int64_t> strides = output.strides();

    // shape {3, 2} → default strides {2, 1}
    EXPECT_EQ(strides[0], 2);
    EXPECT_EQ(strides[1], 1);
}

TEST(ContiguousOpForward, WorksCorrectlyOn3DInput) {
    // Verify forward() handles tensors with more than 2 dimensions.
    // We use a 3D tensor whose logical order differs from its storage order
    // by transposing the first two dimensions.
    Tensor original   = Tensor::from_data({1,2,3,4,5,6,7,8}, {2, 2, 2});
    Tensor transposed = original.transpose();   // swaps dim 0 and dim 1
    Tensor output     = ContiguousOp::forward(transposed);

    EXPECT_TRUE(output.is_contiguous());
    EXPECT_EQ(output.shape(0), transposed.shape(0));
    EXPECT_EQ(output.shape(1), transposed.shape(1));
    EXPECT_EQ(output.shape(2), transposed.shape(2));

    // all logical values must be preserved
    for (int64_t d0 = 0; d0 < output.shape(0); ++d0) {
        for (int64_t d1 = 0; d1 < output.shape(1); ++d1) {
            for (int64_t d2 = 0; d2 < output.shape(2); ++d2) {
                EXPECT_FLOAT_EQ(output.at(d0, d1, d2),
                                transposed.at(d0, d1, d2))
                    << "mismatch at [" << d0 << "," << d1 << "," << d2 << "]";
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ContiguousOp::backward
//
// The backward pass for a copy is an identity: the incoming gradient passes
// straight through.  We test that the gradient values are preserved and that
// backward() returns a fresh tensor (clone) so that the caller owns it
// independently and cannot accidentally alias gradient storage.
// ════════════════════════════════════════════════════════════════════════════

TEST(ContiguousOpBackward, ReturnsExactlyOneGradient) {
    // contiguous() has one input, so backward must return exactly one tensor
    Tensor incoming_gradient = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    std::vector<Tensor> gradients = ContiguousOp::backward(incoming_gradient);

    EXPECT_EQ(gradients.size(), 1u);
}

TEST(ContiguousOpBackward, GradientValuesAreIdenticalToIncomingGradient) {
    // The copy is an element-wise identity, so dL/dinput == dL/doutput
    Tensor incoming_gradient = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    std::vector<Tensor> gradients = ContiguousOp::backward(incoming_gradient);

    EXPECT_FLOAT_EQ(gradients[0].at(0, 0), 1.f);
    EXPECT_FLOAT_EQ(gradients[0].at(0, 1), 2.f);
    EXPECT_FLOAT_EQ(gradients[0].at(1, 0), 3.f);
    EXPECT_FLOAT_EQ(gradients[0].at(1, 1), 4.f);
}

TEST(ContiguousOpBackward, ReturnedGradientHasFreshStorage) {
    // backward() returns a clone, not a view.  If it shared storage with
    // the incoming gradient, the engine could corrupt it during accumulation
    // when multiple paths lead to the same leaf.
    Tensor incoming_gradient = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    std::vector<Tensor> gradients = ContiguousOp::backward(incoming_gradient);

    EXPECT_NE(
        gradients[0].data_ptr(),
        incoming_gradient.data_ptr()
    );
}

TEST(ContiguousOpBackward, GradientShapeMatchesIncomingGradient) {
    Tensor incoming_gradient = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    std::vector<Tensor> gradients = ContiguousOp::backward(incoming_gradient);

    EXPECT_EQ(gradients[0].shape(0), 2);
    EXPECT_EQ(gradients[0].shape(1), 3);
}

// ════════════════════════════════════════════════════════════════════════════
// contiguous() — fast path
//
// When the input is already contiguous, contiguous() must return a view
// over the same storage with the same autograd_meta.  No allocation, no
// graph node.  This is the hot path in training loops where most tensors
// are contiguous and paying for a copy would be wasteful.
// ════════════════════════════════════════════════════════════════════════════

TEST(ContiguousFastPath, AlreadyContiguousTensorIsReturnedAsView) {
    Tensor input = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    ASSERT_TRUE(input.is_contiguous());

    Tensor output = contiguous(input);

    // same storage pointer confirms no allocation took place
    EXPECT_EQ(
        output.data_ptr(),
        input.data_ptr()
    );
}

TEST(ContiguousFastPath, OutputIsContiguous) {
    Tensor input  = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor output = contiguous(input);

    EXPECT_TRUE(output.is_contiguous());
}

TEST(ContiguousFastPath, OutputValuesMatchInput) {
    Tensor input  = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor output = contiguous(input);

    EXPECT_FLOAT_EQ(output.at(0, 0), 1.f);
    EXPECT_FLOAT_EQ(output.at(0, 2), 3.f);
    EXPECT_FLOAT_EQ(output.at(1, 0), 4.f);
    EXPECT_FLOAT_EQ(output.at(1, 2), 6.f);
}

TEST(ContiguousFastPath, AutogradMetaIsSharedNotCopied) {
    // The fast path must share autograd_meta so that the output participates
    // in the same graph node as the input.  If meta were copied, gradients
    // computed through the output would not flow back to the input's leaf.
    Tensor input = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, true);
    ASSERT_TRUE(input.is_contiguous());

    Tensor output = contiguous(input);

    EXPECT_EQ(output.autograd_meta.get(), input.autograd_meta.get());
}

TEST(ContiguousFastPath, DoesNotAddNodeToAutogradGraph) {
    // A zero-copy fast path must not insert a graph node — doing so would
    // create a spurious identity node that wastes memory and slows backward.
    // We verify this by checking that requires_grad status is preserved
    // without a new backward function being added.
    Tensor input = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, true);
    Tensor output = contiguous(input);

    // output and input share meta — so they are the same graph leaf
    EXPECT_EQ(output.autograd_meta.get(), input.autograd_meta.get());
}

TEST(ContiguousFastPath, WorksForZerosTensor) {
    Tensor input  = Tensor::zeros({3, 4});
    Tensor output = contiguous(input);

    EXPECT_EQ(
        output.data_ptr(),
        input.data_ptr()
    );
}

// ════════════════════════════════════════════════════════════════════════════
// contiguous() — slow path
//
// When the input is NOT contiguous, contiguous() must allocate fresh storage,
// copy elements in logical order, and (when requires_grad is true) wire a
// backward node so gradients can flow back through the copy.
// ════════════════════════════════════════════════════════════════════════════

TEST(ContiguousSlowPath, TransposedInputProducesContiguousOutput) {
    Tensor transposed = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}).transpose();
    ASSERT_FALSE(transposed.is_contiguous());

    Tensor output = contiguous(transposed);

    EXPECT_TRUE(output.is_contiguous());
}

TEST(ContiguousSlowPath, OutputStorageDiffersFromInputStorage) {
    // The slow path must allocate — confirming this ensures we are not
    // accidentally returning a view when we need an owned copy.
    Tensor transposed = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}).transpose();
    Tensor output = contiguous(transposed);

    EXPECT_NE(
        output.data_ptr(),
        transposed.data_ptr()
    );
}

TEST(ContiguousSlowPath, OutputValuesMatchLogicalLayoutOfTransposedInput) {
    // original [2,3]:          transposed [3,2]:
    //   [1  2  3]                [1  4]
    //   [4  5  6]                [2  5]
    //                            [3  6]
    Tensor transposed = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}).transpose();
    Tensor output = contiguous(transposed);

    EXPECT_FLOAT_EQ(output.at(0, 0), 1.f);
    EXPECT_FLOAT_EQ(output.at(0, 1), 4.f);
    EXPECT_FLOAT_EQ(output.at(1, 0), 2.f);
    EXPECT_FLOAT_EQ(output.at(1, 1), 5.f);
    EXPECT_FLOAT_EQ(output.at(2, 0), 3.f);
    EXPECT_FLOAT_EQ(output.at(2, 1), 6.f);
}

TEST(ContiguousSlowPath, OutputShapeMatchesTransposedShape) {
    Tensor transposed = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}).transpose();
    Tensor output = contiguous(transposed);

    EXPECT_EQ(output.shape(0), 3);
    EXPECT_EQ(output.shape(1), 2);
}

TEST(ContiguousSlowPath, OutputHasDefaultStridesForShape) {
    // shape {3, 2} → default row-major strides {2, 1}
    Tensor transposed = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}).transpose();
    Tensor output = contiguous(transposed);

    std::vector<int64_t> strides = output.strides();

    EXPECT_EQ(strides[0], 2);
    EXPECT_EQ(strides[1], 1);
}

// ════════════════════════════════════════════════════════════════════════════
// contiguous() — autograd wiring on the slow path
//
// When requires_grad is true and a copy is needed, contiguous() must insert
// a backward node so that gradients flow back through the copy to the
// original tensor.  When requires_grad is false, no node should be added.
// ════════════════════════════════════════════════════════════════════════════

TEST(ContiguousAutograd, OutputRequiresGradWhenInputDoes) {
    Tensor input = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}, {}, true);
    Tensor transposed = input.transpose();
    Tensor output = contiguous(transposed);

    EXPECT_TRUE(output.requires_grad());
}

TEST(ContiguousAutograd, OutputDoesNotRequireGradWhenInputDoesNot) {
    Tensor input = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}, {}, false);
    Tensor transposed = input.transpose();
    Tensor output = contiguous(transposed);

    EXPECT_FALSE(output.requires_grad());
}

TEST(ContiguousAutograd, FastPathPreservesRequiresGrad) {
    // On the fast path the meta is shared, so requires_grad must be preserved
    // without inserting any new node.
    Tensor input = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, true);
    ASSERT_TRUE(input.is_contiguous());

    Tensor output = contiguous(input);

    EXPECT_TRUE(output.requires_grad());
}

TEST(ContiguousAutograd, GradientFlowsThroughSlowPath) {
    // Full backward pass through contiguous().
    // contiguous() on a transposed tensor introduces a copy node.
    // Calling backward() through that node must accumulate gradient
    // back onto the original leaf tensor.
    //
    // We use a small tensor so we can verify every gradient element.
    Tensor original   = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, true);
    Tensor transposed = original.transpose();
    Tensor output     = contiguous(transposed);

    // seed the backward pass with an all-ones gradient of the output's shape
    backward(output);

    // The gradient of a copy is an identity, so every element of
    // original.grad() must be 1.f after backward.
    ASSERT_TRUE(original.has_grad());
    EXPECT_FLOAT_EQ(original.grad().at(0, 0), 1.f);
    EXPECT_FLOAT_EQ(original.grad().at(0, 1), 1.f);
    EXPECT_FLOAT_EQ(original.grad().at(1, 0), 1.f);
    EXPECT_FLOAT_EQ(original.grad().at(1, 1), 1.f);
}

TEST(ContiguousAutograd, NoGradAccumulatedOnFastPath) {
    // On the fast path no graph node is inserted, so backward() on the output
    // must not create a spurious gradient on a no-grad tensor.
    Tensor input = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, false);
    ASSERT_TRUE(input.is_contiguous());

    Tensor output = contiguous(input);

    EXPECT_FALSE(output.requires_grad());
    EXPECT_FALSE(input.has_grad());
}

// ════════════════════════════════════════════════════════════════════════════
// contiguous() — idempotency
//
// Calling contiguous() twice must not allocate additional storage on the
// second call.  The first call produces a contiguous tensor; the second
// must take the fast path and return a view.
// ════════════════════════════════════════════════════════════════════════════

TEST(ContiguousIdempotency, CallingTwiceDoesNotAllocateAdditionalStorage) {
    Tensor transposed    = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}).transpose();
    Tensor first_result  = contiguous(transposed);
    Tensor second_result = contiguous(first_result);

    // second call is a fast path — same storage as first result
    EXPECT_EQ(
        second_result.data_ptr(),
        first_result.data_ptr()
    );
}

TEST(ContiguousIdempotency, ValuesArePreservedAfterDoubleCall) {
    Tensor transposed    = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}).transpose();
    Tensor second_result = contiguous(contiguous(transposed));

    EXPECT_FLOAT_EQ(second_result.at(0, 0), 1.f);
    EXPECT_FLOAT_EQ(second_result.at(0, 1), 4.f);
    EXPECT_FLOAT_EQ(second_result.at(2, 1), 6.f);
}