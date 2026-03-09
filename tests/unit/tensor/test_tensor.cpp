// tests/test_tensor.cpp
//
// Unit tests for the public Tensor API.
//
// Tensor is the user-facing wrapper around TensorImpl.  It owns the
// implementation pointer, exposes factory functions, and provides the
// element-access, shape, clone, transpose, contiguity, and autograd-metadata
// interfaces that every other component in the library relies on.
//
// Test organisation (in file order):
//   1.  Tensor::zeros          — factory, shape, ndim, numel, all-zero values
//   2.  Tensor::from_data      — factory, correct values loaded, shape propagated
//   3.  Shape and metadata     — ndim(), shape(), numel() for 1-D through 4-D
//   4.  Element read/write     — at() read, at() write, boundary and interior
//   5.  Independence           — two tensors from the same data do not share storage
//   6.  clone()                — value fidelity, storage independence, shape/ndim
//   7.  transpose()            — shape swap, zero-copy storage, value mapping,
//                                double-transpose identity, mutation propagation
//   8.  is_contiguous()        — fresh tensors, transposed tensors,
//                                double-transposed, clone of non-contiguous
//   9.  Autograd metadata      — requires_grad flag, has_grad before backward,
//                                independence between tensors
//  10.  Internal strides       — verify implementation->strides directly
//                                (documents the contract TensorImpl depends on)
//
// What is NOT tested here:
//   - default_strides() and TensorImpl::at()  →  test_tensor_impl.cpp
//   - matmul(), backward(), Linear, SGD       →  their own test files
//
// All tests follow the project conventions established in test_tensor_impl.cpp:
//   - no `auto` — every type is spelled out explicitly
//   - descriptive variable names that read as sentences
//   - comments explain *why* a check is needed, not just what it does

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>
#include "tensor/tensor.h"

// ════════════════════════════════════════════════════════════════════════════
// 1.  Tensor::zeros
//
// zeros() must allocate a contiguous tensor of the requested shape whose
// every element is exactly 0.f.  We check shape metadata and values
// separately so a failure tells us whether the problem is in the allocation
// bookkeeping or in the initialisation loop.
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorZeros, TwoDimensionalShapeIsRecordedCorrectly) {
    Tensor tensor = Tensor::zeros({2, 3});

    EXPECT_EQ(tensor.ndim(),    2);
    EXPECT_EQ(tensor.shape(0),  2);
    EXPECT_EQ(tensor.shape(1),  3);
    EXPECT_EQ(tensor.numel(),   6);
}

TEST(TensorZeros, ThreeDimensionalShapeIsRecordedCorrectly) {
    Tensor tensor = Tensor::zeros({2, 3, 4});

    EXPECT_EQ(tensor.ndim(),    3);
    EXPECT_EQ(tensor.shape(0),  2);
    EXPECT_EQ(tensor.shape(1),  3);
    EXPECT_EQ(tensor.shape(2),  4);
    EXPECT_EQ(tensor.numel(),  24);
}

TEST(TensorZeros, FourDimensionalShapeIsRecordedCorrectly) {
    Tensor tensor = Tensor::zeros({2, 3, 4, 5});

    EXPECT_EQ(tensor.ndim(),    4);
    EXPECT_EQ(tensor.numel(), 120);
}

TEST(TensorZeros, AllElementsAreZero2D) {
    Tensor tensor = Tensor::zeros({3, 4});

    for (int64_t row = 0; row < 3; ++row) {
        for (int64_t col = 0; col < 4; ++col) {
            EXPECT_FLOAT_EQ(tensor.at(row, col), 0.f)
                << "expected 0 at [" << row << ", " << col << "]";
        }
    }
}

TEST(TensorZeros, AllElementsAreZero3D) {
    Tensor tensor = Tensor::zeros({2, 3, 4});

    for (int64_t d0 = 0; d0 < 2; ++d0) {
        for (int64_t d1 = 0; d1 < 3; ++d1) {
            for (int64_t d2 = 0; d2 < 4; ++d2) {
                EXPECT_FLOAT_EQ(tensor.at(d0, d1, d2), 0.f)
                    << "expected 0 at [" << d0 << ", " << d1 << ", " << d2 << "]";
            }
        }
    }
}

TEST(TensorZeros, DoesNotRequireGradByDefault) {
    // zeros() without the requires_grad flag must produce a leaf tensor
    // that is not part of the autograd graph.
    Tensor tensor = Tensor::zeros({2, 2});
    EXPECT_FALSE(tensor.requires_grad());
}

TEST(TensorZeros, RequiresGradWhenFlagIsTrue) {
    Tensor tensor = Tensor::zeros({2, 2}, true);
    EXPECT_TRUE(tensor.requires_grad());
}

// ════════════════════════════════════════════════════════════════════════════
// 2.  Tensor::from_data
//
// from_data() must copy the supplied initialiser list into fresh storage
// and lay it out in row-major order for the given shape.
//
// We test value correctness at every corner and one interior position for
// each dimensionality.  Corner tests catch row/column swap bugs; interior
// tests catch stride-computation bugs that would only show up away from
// the origin.
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorFromData, TwoDimensionalShapeIsRecordedCorrectly) {
    Tensor tensor = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});

    EXPECT_EQ(tensor.ndim(),   2);
    EXPECT_EQ(tensor.shape(0), 2);
    EXPECT_EQ(tensor.shape(1), 3);
    EXPECT_EQ(tensor.numel(),  6);
}

TEST(TensorFromData, TwoDimensionalCornerValues) {
    // Layout (row-major):
    //   [1  2  3]
    //   [4  5  6]
    Tensor tensor = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});

    EXPECT_FLOAT_EQ(tensor.at(0, 0), 1.f);   // top-left
    EXPECT_FLOAT_EQ(tensor.at(0, 2), 3.f);   // top-right
    EXPECT_FLOAT_EQ(tensor.at(1, 0), 4.f);   // bottom-left
    EXPECT_FLOAT_EQ(tensor.at(1, 2), 6.f);   // bottom-right
}

TEST(TensorFromData, TwoDimensionalInteriorValue) {
    Tensor tensor = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});

    // [0,1] is the second column of the first row — value 2
    EXPECT_FLOAT_EQ(tensor.at(0, 1), 2.f);
    // [1,1] is the centre element — value 5
    EXPECT_FLOAT_EQ(tensor.at(1, 1), 5.f);
}

TEST(TensorFromData, ThreeDimensionalShapeIsRecordedCorrectly) {
    Tensor tensor = Tensor::from_data({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});

    EXPECT_EQ(tensor.ndim(),   3);
    EXPECT_EQ(tensor.shape(0), 2);
    EXPECT_EQ(tensor.shape(1), 2);
    EXPECT_EQ(tensor.shape(2), 2);
    EXPECT_EQ(tensor.numel(),  8);
}

TEST(TensorFromData, ThreeDimensionalCornerValues) {
    // Layout (row-major, two 2×2 slabs):
    //   slab 0:  [1  2]    slab 1:  [5  6]
    //            [3  4]             [7  8]
    Tensor tensor = Tensor::from_data({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});

    EXPECT_FLOAT_EQ(tensor.at(0, 0, 0), 1.f);   // first element of slab 0
    EXPECT_FLOAT_EQ(tensor.at(0, 1, 1), 4.f);   // last element of slab 0
    EXPECT_FLOAT_EQ(tensor.at(1, 0, 0), 5.f);   // first element of slab 1
    EXPECT_FLOAT_EQ(tensor.at(1, 1, 1), 8.f);   // last element overall
}

TEST(TensorFromData, DoesNotRequireGradByDefault) {
    Tensor tensor = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    EXPECT_FALSE(tensor.requires_grad());
}

TEST(TensorFromData, RequiresGradWhenFlagIsSet) {
    // The empty strides argument {} tells from_data to compute default strides.
    Tensor tensor = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, true);
    EXPECT_TRUE(tensor.requires_grad());
}

// ════════════════════════════════════════════════════════════════════════════
// 3.  Shape and metadata accessors
//
// ndim(), shape(), and numel() are used everywhere in the codebase to guard
// against mismatched operands.  We verify that the values they return are
// consistent with each other and with the shape passed to the factory.
//
// numel() must equal the product of all dimensions.  We check this invariant
// directly because a bug that returns storage.size() instead of the product
// of the logical shape would be silently wrong for non-contiguous views.
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorMetadata, NumelEqualsProductOfDimensions2D) {
    Tensor tensor = Tensor::zeros({5, 7});
    EXPECT_EQ(tensor.numel(), 5 * 7);
}

TEST(TensorMetadata, NumelEqualsProductOfDimensions3D) {
    Tensor tensor = Tensor::zeros({2, 5, 7});
    EXPECT_EQ(tensor.numel(), 2 * 5 * 7);
}

TEST(TensorMetadata, NumelEqualsProductOfDimensions4D) {
    Tensor tensor = Tensor::zeros({2, 3, 5, 7});
    EXPECT_EQ(tensor.numel(), 2 * 3 * 5 * 7);
}

TEST(TensorMetadata, ShapeAccessorMatchesConstructedShape) {
    // Verify each dimension individually so a failure names the offending axis.
    Tensor tensor = Tensor::zeros({6, 7, 8, 9});

    EXPECT_EQ(tensor.shape(0), 6);
    EXPECT_EQ(tensor.shape(1), 7);
    EXPECT_EQ(tensor.shape(2), 8);
    EXPECT_EQ(tensor.shape(3), 9);
}

TEST(TensorMetadata, NdimMatchesNumberOfAxes) {
    EXPECT_EQ(Tensor::zeros({5}).ndim(),           1);
    EXPECT_EQ(Tensor::zeros({3, 4}).ndim(),        2);
    EXPECT_EQ(Tensor::zeros({2, 3, 4}).ndim(),     3);
    EXPECT_EQ(Tensor::zeros({2, 3, 4, 5}).ndim(),  4);
}

// ════════════════════════════════════════════════════════════════════════════
// 4.  Element read / write  (at())
//
// at() is the single gateway to tensor data.  It must:
//   - return the correct value for every valid multi-index
//   - return a reference so assignments go to the right storage location
//   - not disturb adjacent elements when writing
//
// We test boundary and interior positions, and we cross-check writes by
// reading back through at() *and* through the raw storage pointer so that
// a symmetric bug (wrong index on both read and write) cannot hide.
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorReadWrite, ReadFromZerosTensor2D) {
    Tensor tensor = Tensor::zeros({3, 3});

    EXPECT_FLOAT_EQ(tensor.at(0, 0), 0.f);
    EXPECT_FLOAT_EQ(tensor.at(2, 2), 0.f);
}

TEST(TensorReadWrite, WriteAndReadBack2D) {
    Tensor tensor = Tensor::zeros({3, 3});
    tensor.at(0, 1) = 7.f;
    EXPECT_FLOAT_EQ(tensor.at(0, 1), 7.f);
}

TEST(TensorReadWrite, WriteAndReadBack3D) {
    Tensor tensor = Tensor::zeros({2, 3, 4});
    tensor.at(1, 2, 3) = 42.f;
    EXPECT_FLOAT_EQ(tensor.at(1, 2, 3), 42.f);
}

TEST(TensorReadWrite, WriteAndReadBack4D) {
    Tensor tensor = Tensor::zeros({2, 3, 4, 5});
    tensor.at(1, 2, 3, 4) = 99.f;
    EXPECT_FLOAT_EQ(tensor.at(1, 2, 3, 4), 99.f);
}

TEST(TensorReadWrite, WriteDoesNotCorruptNeighbouringElements) {
    // Write to a single interior element and check that the elements
    // immediately before and after it in storage are untouched.
    // This would catch an off-by-one error in the stride arithmetic.
    Tensor tensor = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    tensor.at(0, 1) = 99.f;   // middle of the first row

    EXPECT_FLOAT_EQ(tensor.at(0, 0), 1.f);   // element before
    EXPECT_FLOAT_EQ(tensor.at(0, 2), 3.f);   // element after
}

TEST(TensorReadWrite, WriteIsVisibleThroughRawStoragePointer) {
    // Cross-check: the write must land at the correct flat offset in the
    // underlying storage, not just at whichever offset at() happens to read.
    // For shape {2,3} with strides {3,1}, at({1,0}) → flat index 3.
    Tensor tensor = Tensor::zeros({2, 3});
    tensor.at(1, 0) = 55.f;

    EXPECT_FLOAT_EQ(tensor.implementation->storage->ptr()[3], 55.f);
}

TEST(TensorReadWrite, MultipleDistinctWritesDoNotInterfere) {
    // Write to several elements and verify each is independently retained.
    Tensor tensor = Tensor::zeros({3, 3});
    tensor.at(0, 0) = 10.f;
    tensor.at(1, 1) = 20.f;
    tensor.at(2, 2) = 30.f;

    EXPECT_FLOAT_EQ(tensor.at(0, 0), 10.f);
    EXPECT_FLOAT_EQ(tensor.at(1, 1), 20.f);
    EXPECT_FLOAT_EQ(tensor.at(2, 2), 30.f);
}

// ════════════════════════════════════════════════════════════════════════════
// 5.  Storage independence between separate tensors
//
// Two tensors created from the same data initialiser must own separate
// storage allocations.  If they shared memory, writing to one would silently
// corrupt the other — an extremely difficult bug to track down in training
// loops where tensors are reused across epochs.
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorIndependence, SeparateTensorsHaveDifferentStoragePointers) {
    Tensor tensor_a = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    Tensor tensor_b = Tensor::from_data({1, 2, 3, 4}, {2, 2});

    EXPECT_NE(
        tensor_a.implementation->storage.get(),
        tensor_b.implementation->storage.get()
    );
}

TEST(TensorIndependence, WritingToOneTensorDoesNotAffectAnother) {
    Tensor tensor_a = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    Tensor tensor_b = Tensor::from_data({1, 2, 3, 4}, {2, 2});

    tensor_a.at(0, 0) = 99.f;

    EXPECT_FLOAT_EQ(tensor_b.at(0, 0), 1.f);
}

TEST(TensorIndependence, TwoZerosTensorsHaveDifferentStoragePointers) {
    // zeros() must allocate fresh storage each call, not return a cached tensor.
    Tensor tensor_a = Tensor::zeros({3, 3});
    Tensor tensor_b = Tensor::zeros({3, 3});

    EXPECT_NE(
        tensor_a.implementation->storage.get(),
        tensor_b.implementation->storage.get()
    );
}

// ════════════════════════════════════════════════════════════════════════════
// 6.  clone()
//
// clone() must produce an exact value copy with independent storage.  It is
// the explicit way to materialise a new allocation — critical when you need
// to save activations before an in-place update.
//
// We test:
//   - value fidelity (every element matches)
//   - storage independence (pointer differs, mutation does not propagate)
//   - shape and ndim are preserved
//   - cloning a non-contiguous (transposed) tensor produces a contiguous copy
//     with the correct logical values
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorClone, ClonedTensorHasSameValues2D) {
    Tensor original = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor cloned   = original.clone();

    EXPECT_FLOAT_EQ(cloned.at(0, 0), 1.f);
    EXPECT_FLOAT_EQ(cloned.at(0, 2), 3.f);
    EXPECT_FLOAT_EQ(cloned.at(1, 0), 4.f);
    EXPECT_FLOAT_EQ(cloned.at(1, 2), 6.f);
}

TEST(TensorClone, ClonedTensorHasSameShape) {
    Tensor original = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor cloned   = original.clone();

    EXPECT_EQ(cloned.ndim(),   original.ndim());
    EXPECT_EQ(cloned.shape(0), original.shape(0));
    EXPECT_EQ(cloned.shape(1), original.shape(1));
    EXPECT_EQ(cloned.numel(),  original.numel());
}

TEST(TensorClone, ClonedTensorHasDifferentStoragePointer) {
    // If storage were shared, the next test (mutation) would be vacuous.
    // Checking the pointer directly makes the independence guarantee explicit.
    Tensor original = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor cloned   = original.clone();

    EXPECT_NE(
        cloned.implementation->storage.get(),
        original.implementation->storage.get()
    );
}

TEST(TensorClone, MutatingCloneDoesNotAffectOriginal) {
    Tensor original = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor cloned   = original.clone();

    cloned.at(0, 0) = 99.f;

    EXPECT_FLOAT_EQ(original.at(0, 0), 1.f);
}

TEST(TensorClone, MutatingOriginalDoesNotAffectClone) {
    Tensor original = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor cloned   = original.clone();

    original.at(0, 0) = 99.f;

    EXPECT_FLOAT_EQ(cloned.at(0, 0), 1.f);
}

TEST(TensorClone, ClonePreservesNdimFor3D) {
    Tensor original = Tensor::from_data({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
    Tensor cloned   = original.clone();

    EXPECT_EQ(cloned.ndim(),      3);
    EXPECT_FLOAT_EQ(cloned.at(1, 1, 1), 8.f);
}

TEST(TensorClone, CloneOfTransposedTensorHasCorrectLogicalValues) {
    // Cloning a non-contiguous tensor must materialise the logical view,
    // not just copy the raw storage bytes.  The resulting tensor must
    // read the same values as the transposed view, but laid out contiguously.
    //
    // original [2,3]:          transposed [3,2]:
    //   [1  2  3]                [1  4]
    //   [4  5  6]                [2  5]
    //                            [3  6]
    Tensor original    = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor transposed  = original.transpose();
    Tensor cloned      = transposed.clone();

    EXPECT_EQ(cloned.shape(0), 3);
    EXPECT_EQ(cloned.shape(1), 2);

    EXPECT_FLOAT_EQ(cloned.at(0, 0), 1.f);
    EXPECT_FLOAT_EQ(cloned.at(0, 1), 4.f);
    EXPECT_FLOAT_EQ(cloned.at(1, 0), 2.f);
    EXPECT_FLOAT_EQ(cloned.at(1, 1), 5.f);
    EXPECT_FLOAT_EQ(cloned.at(2, 0), 3.f);
    EXPECT_FLOAT_EQ(cloned.at(2, 1), 6.f);
}

TEST(TensorClone, CloneOfTransposedTensorIsContiguous) {
    // After cloning, the copy must be in default row-major order regardless
    // of how the source tensor was laid out.  Operations like matmul that
    // assume contiguity depend on this.
    Tensor original   = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor transposed = original.transpose();
    Tensor cloned     = transposed.clone();

    EXPECT_TRUE(cloned.is_contiguous());
}

// ════════════════════════════════════════════════════════════════════════════
// 7.  transpose()
//
// transpose() must return a zero-copy view with swapped shape and strides.
// "Zero-copy" means the underlying storage pointer is shared — no new
// allocation and no data movement.  Mutations through the transposed view
// must be visible in the original and vice versa.
//
// We test:
//   - shape dimensions are swapped
//   - storage pointer is identical (zero-copy guarantee)
//   - logical values are correctly remapped
//   - double transpose restores original shape and contiguity
//   - a write through the transposed view modifies the original storage
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorTranspose, ShapeIsSwapped) {
    Tensor original   = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor transposed = original.transpose();

    EXPECT_EQ(transposed.shape(0), 3);   // was 2
    EXPECT_EQ(transposed.shape(1), 2);   // was 3
}

TEST(TensorTranspose, NdimIsPreserved) {
    Tensor original   = Tensor::zeros({2, 3});
    Tensor transposed = original.transpose();

    EXPECT_EQ(transposed.ndim(), 2);
}

TEST(TensorTranspose, NumelIsPreserved) {
    Tensor original   = Tensor::zeros({2, 3});
    Tensor transposed = original.transpose();

    EXPECT_EQ(transposed.numel(), 6);
}

TEST(TensorTranspose, StoragePointerIsSharedZeroCopy) {
    // The whole point of a transposed view is that it reuses existing memory.
    // If this fails it means transpose() allocated a new buffer, which would
    // break the mutation-propagation contract below.
    Tensor original   = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor transposed = original.transpose();

    EXPECT_EQ(
        transposed.implementation->storage.get(),
        original.implementation->storage.get()
    );
}

TEST(TensorTranspose, ElementValuesAreRemappedCorrectly) {
    // original [2,3]:          transposed [3,2]:
    //   [1  2  3]                [1  4]
    //   [4  5  6]                [2  5]
    //                            [3  6]
    Tensor original   = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor transposed = original.transpose();

    EXPECT_FLOAT_EQ(transposed.at(0, 0), 1.f);
    EXPECT_FLOAT_EQ(transposed.at(0, 1), 4.f);
    EXPECT_FLOAT_EQ(transposed.at(1, 0), 2.f);
    EXPECT_FLOAT_EQ(transposed.at(1, 1), 5.f);
    EXPECT_FLOAT_EQ(transposed.at(2, 0), 3.f);
    EXPECT_FLOAT_EQ(transposed.at(2, 1), 6.f);
}

TEST(TensorTranspose, DoubleTransposeRestoresOriginalShape) {
    Tensor original          = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor double_transposed = original.transpose().transpose();

    EXPECT_EQ(double_transposed.shape(0), 2);
    EXPECT_EQ(double_transposed.shape(1), 3);
}

TEST(TensorTranspose, DoubleTransposeRestoresOriginalValues) {
    Tensor original          = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor double_transposed = original.transpose().transpose();

    EXPECT_FLOAT_EQ(double_transposed.at(0, 0), 1.f);
    EXPECT_FLOAT_EQ(double_transposed.at(0, 2), 3.f);
    EXPECT_FLOAT_EQ(double_transposed.at(1, 0), 4.f);
    EXPECT_FLOAT_EQ(double_transposed.at(1, 2), 6.f);
}

TEST(TensorTranspose, WriteThroughTransposedViewAffectsOriginal) {
    // Because transposed shares storage with original, a write through the
    // transposed view must immediately be visible in the original at the
    // corresponding logical position.
    //
    // transposed[1, 0] corresponds to original[0, 1] (column and row swapped).
    Tensor original   = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor transposed = original.transpose();

    transposed.at(1, 0) = 99.f;

    EXPECT_FLOAT_EQ(original.at(0, 1), 99.f);
}

TEST(TensorTranspose, WriteToOriginalAffectsTransposedView) {
    // The reverse direction: original write must propagate into the view.
    Tensor original   = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor transposed = original.transpose();

    original.at(1, 2) = 77.f;   // bottom-right of original

    // In transposed that position becomes [2, 1] (last row, second column)
    EXPECT_FLOAT_EQ(transposed.at(2, 1), 77.f);
}

// ════════════════════════════════════════════════════════════════════════════
// 8.  is_contiguous()
//
// A tensor is contiguous when its strides match the default row-major layout
// for its shape.  This matters because operations like matmul may need to
// call clone() on non-contiguous inputs before processing them.
//
// Key cases:
//   - freshly created tensors are always contiguous
//   - transpose() produces a non-contiguous view (strides are swapped)
//   - double-transpose restores contiguity (strides match shape again)
//   - clone() of a non-contiguous tensor is always contiguous
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorIsContiguous, ZerosTensorIsContiguous) {
    EXPECT_TRUE(Tensor::zeros({3, 4}).is_contiguous());
}

TEST(TensorIsContiguous, FromDataTensorIsContiguous) {
    Tensor tensor = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    EXPECT_TRUE(tensor.is_contiguous());
}

TEST(TensorIsContiguous, ThreeDimZerosTensorIsContiguous) {
    EXPECT_TRUE(Tensor::zeros({2, 3, 4}).is_contiguous());
}

TEST(TensorIsContiguous, TransposedTensorIsNotContiguous) {
    // After transposing, the strides are {1, 3} but default strides for
    // shape {3, 2} would be {2, 1} — they disagree, so not contiguous.
    Tensor original   = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor transposed = original.transpose();

    EXPECT_FALSE(transposed.is_contiguous());
}

TEST(TensorIsContiguous, DoubleTransposedTensorIsContiguous) {
    // Transposing twice returns strides {3, 1} for shape {2, 3}, which
    // matches default row-major strides — so it is contiguous again.
    Tensor original          = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor double_transposed = original.transpose().transpose();

    EXPECT_TRUE(double_transposed.is_contiguous());
}

TEST(TensorIsContiguous, CloneOfTransposedTensorIsContiguous) {
    // clone() materialises a fresh row-major copy, so it must always be
    // contiguous regardless of the source tensor's layout.
    Tensor transposed = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}).transpose();
    Tensor cloned     = transposed.clone();

    EXPECT_TRUE(cloned.is_contiguous());
}

// ════════════════════════════════════════════════════════════════════════════
// 9.  Autograd metadata  (requires_grad / has_grad)
//
// requires_grad marks tensors that participate in gradient computation.
// has_grad becomes true only after backward() has accumulated a gradient.
// These two flags are independent: requires_grad is set at construction,
// has_grad is set later by the autograd engine.
//
// We do not call backward() here — that is covered in test_autograd.cpp.
// Here we only verify the metadata state before any backward pass.
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorAutogradMeta, RequiresGradIsFalseByDefault) {
    Tensor tensor = Tensor::zeros({2, 2});
    EXPECT_FALSE(tensor.requires_grad());
}

TEST(TensorAutogradMeta, RequiresGradIsTrueWhenRequested) {
    Tensor tensor = Tensor::zeros({2, 2}, true);
    EXPECT_TRUE(tensor.requires_grad());
}

TEST(TensorAutogradMeta, HasGradIsFalseBeforeBackward) {
    // A leaf tensor with requires_grad=true must NOT have a gradient
    // until backward() is called.  Returning a spurious zero gradient
    // would corrupt gradient accumulation in multi-step training loops.
    Tensor tensor = Tensor::zeros({2, 2}, true);
    EXPECT_FALSE(tensor.has_grad());
}

TEST(TensorAutogradMeta, HasGradIsFalseForNonGradTensor) {
    Tensor tensor = Tensor::zeros({2, 2}, false);
    EXPECT_FALSE(tensor.has_grad());
}

TEST(TensorAutogradMeta, RequiresGradFromDataWithFlag) {
    Tensor tensor = Tensor::from_data({1, 2, 3, 4}, {2, 2}, {}, true);
    EXPECT_TRUE(tensor.requires_grad());
    EXPECT_FALSE(tensor.has_grad());
}

TEST(TensorAutogradMeta, FlagsAreIndependentBetweenTensors) {
    // The requires_grad flag must be per-tensor, not a global or shared state.
    // If it were shared, setting it on one tensor would corrupt the other.
    Tensor grad_tensor    = Tensor::zeros({2, 2}, true);
    Tensor no_grad_tensor = Tensor::zeros({2, 2}, false);

    EXPECT_TRUE(grad_tensor.requires_grad());
    EXPECT_FALSE(no_grad_tensor.requires_grad());
}

TEST(TensorAutogradMeta, ClonedTensorDoesNotInheritRequiresGrad) {
    // clone() creates a detached copy — it should NOT carry over the
    // autograd graph membership of its source.  If it did, an accidental
    // clone inside a loss function would silently add a leaf to the graph.
    Tensor original = Tensor::zeros({2, 2}, true);
    Tensor cloned   = original.clone();

    EXPECT_FALSE(cloned.requires_grad());
}

// ════════════════════════════════════════════════════════════════════════════
// 10.  Internal strides  (implementation->strides)
//
// Strides are the bridge between Tensor's logical shape and TensorImpl's
// element-access arithmetic.  We inspect them directly here so that if a
// higher-level test (e.g. a wrong matmul result) fails, we can immediately
// rule out a stride bug by running this suite.
//
// We also verify that transpose() correctly swaps the strides (not just the
// shape), since the shape swap alone would not change which storage element
// at() resolves to.
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorInternalStrides, DefaultStridesFor2DShape) {
    // shape {3, 4} → row-major strides must be {4, 1}
    Tensor tensor = Tensor::zeros({3, 4});

    EXPECT_EQ(tensor.implementation->strides[0], 4);
    EXPECT_EQ(tensor.implementation->strides[1], 1);
}

TEST(TensorInternalStrides, DefaultStridesFor3DShape) {
    // shape {2, 3, 4} → strides must be {12, 4, 1}
    Tensor tensor = Tensor::zeros({2, 3, 4});

    EXPECT_EQ(tensor.implementation->strides[0], 12);
    EXPECT_EQ(tensor.implementation->strides[1],  4);
    EXPECT_EQ(tensor.implementation->strides[2],  1);
}

TEST(TensorInternalStrides, DefaultStridesFor4DShape) {
    // shape {2, 3, 4, 5} → strides must be {60, 20, 5, 1}
    Tensor tensor = Tensor::zeros({2, 3, 4, 5});

    EXPECT_EQ(tensor.implementation->strides[0], 60);
    EXPECT_EQ(tensor.implementation->strides[1], 20);
    EXPECT_EQ(tensor.implementation->strides[2],  5);
    EXPECT_EQ(tensor.implementation->strides[3],  1);
}

TEST(TensorInternalStrides, TransposeSwapsStrides) {
    // Transposing [2, 3] (strides {3, 1}) must produce [3, 2] with strides {1, 3}.
    // If only the shape were swapped the indexing arithmetic would still use
    // the original strides and produce wrong values.
    Tensor original   = Tensor::zeros({2, 3});
    Tensor transposed = original.transpose();

    EXPECT_EQ(transposed.implementation->strides[0], 1);
    EXPECT_EQ(transposed.implementation->strides[1], 3);
}

TEST(TensorInternalStrides, DoubleTransposeRestoresOriginalStrides) {
    Tensor original          = Tensor::zeros({2, 3});
    Tensor double_transposed = original.transpose().transpose();

    EXPECT_EQ(double_transposed.implementation->strides[0], 3);
    EXPECT_EQ(double_transposed.implementation->strides[1], 1);
}

TEST(TensorInternalStrides, CloneOfTransposedTensorHasDefaultStrides) {
    // A clone always materialises a fresh contiguous layout, so its strides
    // must match the default for its shape — {2, 1} for shape {3, 2}.
    Tensor transposed = Tensor::zeros({2, 3}).transpose();   // shape {3,2}
    Tensor cloned     = transposed.clone();

    EXPECT_EQ(cloned.implementation->strides[0], 2);
    EXPECT_EQ(cloned.implementation->strides[1], 1);
}