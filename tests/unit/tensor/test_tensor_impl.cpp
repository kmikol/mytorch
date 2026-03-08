// tests/test_tensor_impl.cpp
//
// Unit tests for:
//   - default_strides()   — computes row-major strides from a shape vector
//   - TensorImpl          — the internal tensor view: shape, strides, element access
//
// These tests sit below Tensor (the public API) and above Storage (raw memory).
// They verify the indexing arithmetic that every higher-level operation depends on.
// If these break, matmul, transpose, autograd — everything breaks with them.

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "tensor/tensor_implementation.h"

// ── test helper ──────────────────────────────────────────────────────────────
//
// Creates a Storage of size n filled with 0.f, 1.f, 2.f, ... (n-1).f
// This gives every element a unique, predictable value so that if the wrong
// element is read we'll see the wrong number rather than a silent 0.
static std::shared_ptr<Storage> make_iota_storage(int n) {
    std::shared_ptr<Storage> storage = std::make_shared<Storage>(n);
    for (int i = 0; i < n; ++i) {
        storage->ptr()[i] = static_cast<float>(i);
    }
    return storage;
}

// ════════════════════════════════════════════════════════════════════════════
// default_strides()
//
// Row-major (C-order) strides: for shape [d0, d1, d2, ...],
//   stride[i] = product of all dimensions after i
//
// Example: shape {2,3,4} → strides {12, 4, 1}
//   - moving along dim-0 skips 3*4 = 12 elements
//   - moving along dim-1 skips 4 elements
//   - moving along dim-2 skips 1 element (contiguous)
//
// This is tested independently because TensorImpl receives pre-computed strides.
// A bug here would silently corrupt every element access in every tensor.
// ════════════════════════════════════════════════════════════════════════════

TEST(DefaultStrides, OneDimensionalShape) {
    // A 1-D tensor is just a flat array.
    // The only stride is 1: each step moves one element forward.
    std::vector<int64_t> strides = default_strides({5});

    ASSERT_EQ(strides.size(), 1u);
    EXPECT_EQ(strides[0], 1);
}

TEST(DefaultStrides, TwoDimensionalShape) {
    // shape {3, 4}: rows of 4 elements.
    // stride[0] = 4  (skip a whole row to move to the next row)
    // stride[1] = 1  (move one element to reach the next column)
    std::vector<int64_t> strides = default_strides({3, 4});

    ASSERT_EQ(strides.size(), 2u);
    EXPECT_EQ(strides[0], 4);
    EXPECT_EQ(strides[1], 1);
}

TEST(DefaultStrides, ThreeDimensionalShape) {
    // shape {2, 3, 4}: two slabs of 3×4 matrices.
    // stride[0] = 3*4 = 12  (skip a whole 3×4 slab)
    // stride[1] = 4         (skip a whole row of 4)
    // stride[2] = 1         (move one element)
    std::vector<int64_t> strides = default_strides({2, 3, 4});

    ASSERT_EQ(strides.size(), 3u);
    EXPECT_EQ(strides[0], 12);
    EXPECT_EQ(strides[1],  4);
    EXPECT_EQ(strides[2],  1);
}

TEST(DefaultStrides, FourDimensionalShape) {
    // shape {2, 3, 4, 5}: typical batch × channels × height × width layout.
    // stride[0] = 3*4*5 = 60
    // stride[1] = 4*5   = 20
    // stride[2] = 5
    // stride[3] = 1
    std::vector<int64_t> strides = default_strides({2, 3, 4, 5});

    ASSERT_EQ(strides.size(), 4u);
    EXPECT_EQ(strides[0], 60);
    EXPECT_EQ(strides[1], 20);
    EXPECT_EQ(strides[2],  5);
    EXPECT_EQ(strides[3],  1);
}

TEST(DefaultStrides, LastStrideIsAlwaysOne) {
    // The innermost (last) dimension is always contiguous — stride 1.
    // This is a fundamental invariant of row-major layout.
    // We check it across several shapes rather than just one to make
    // sure the implementation doesn't accidentally hard-code it for 2D.
    std::vector<std::vector<int64_t>> shapes = {
        {5},
        {3, 4},
        {2, 3, 4},
        {2, 3, 4, 5}
    };

    for (const std::vector<int64_t>& shape : shapes) {
        std::vector<int64_t> strides = default_strides(shape);
        EXPECT_EQ(strides.back(), 1)
            << "last stride must always be 1 for row-major layout";
    }
}

TEST(DefaultStrides, FirstStrideEqualsProductOfRemainingDims) {
    // stride[0] must equal the product of all other dimensions.
    // For shape {2, 5, 3}: stride[0] = 5 * 3 = 15
    // This is the "skip one entire outer slice" rule.
    std::vector<int64_t> strides = default_strides({2, 5, 3});

    EXPECT_EQ(strides[0], 15);
}

TEST(DefaultStrides, AllSizeOneDimensions) {
    // When every dimension is 1, every stride is also 1.
    // There is only one element, so all steps land on the same location.
    std::vector<int64_t> strides = default_strides({1, 1, 1});

    ASSERT_EQ(strides.size(), 3u);
    EXPECT_EQ(strides[0], 1);
    EXPECT_EQ(strides[1], 1);
    EXPECT_EQ(strides[2], 1);
}

// ════════════════════════════════════════════════════════════════════════════
// TensorImpl — construction and metadata
//
// TensorImpl wraps a shared Storage together with shape and stride vectors.
// It does not own the data; it is a *view* over an existing allocation.
// Multiple TensorImpl instances can point to the same Storage (e.g. transpose).
//
// The fixture creates two storages used across many tests so each test gets
// a clean, predictable starting state without repeating the setup code.
// ════════════════════════════════════════════════════════════════════════════

class TensorImplTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 12-element storage backing a logical [3, 4] tensor:
        // flat contents:  0  1  2  3  4  5  6  7  8  9  10  11
        storage_2d = make_iota_storage(12);

        // 24-element storage backing a logical [2, 3, 4] tensor:
        // flat contents:  0  1  2 ... 23
        storage_3d = make_iota_storage(24);
    }

    std::shared_ptr<Storage> storage_2d;
    std::shared_ptr<Storage> storage_3d;
};

TEST_F(TensorImplTest, NdimReportsCorrectRankForTwoDimensions) {
    TensorImpl tensor_impl(storage_2d, {3, 4}, {4, 1});
    EXPECT_EQ(tensor_impl.ndim(), 2);
}

TEST_F(TensorImplTest, NdimReportsCorrectRankForThreeDimensions) {
    TensorImpl tensor_impl(storage_3d, {2, 3, 4}, {12, 4, 1});
    EXPECT_EQ(tensor_impl.ndim(), 3);
}

TEST_F(TensorImplTest, NumelIsProductOfDimensionsForTwoDim) {
    // numel() must return 3*4 = 12, not the size of the underlying storage.
    // They happen to match here, but they can differ for views (e.g. slices).
    TensorImpl tensor_impl(storage_2d, {3, 4}, {4, 1});
    EXPECT_EQ(tensor_impl.numel(), 12);
}

TEST_F(TensorImplTest, NumelIsProductOfDimensionsForThreeDim) {
    TensorImpl tensor_impl(storage_3d, {2, 3, 4}, {12, 4, 1});
    EXPECT_EQ(tensor_impl.numel(), 24);
}

TEST_F(TensorImplTest, StoragePointerMatchesSuppliedStorage) {
    // TensorImpl must hold a reference to the exact storage we gave it,
    // not a copy. This is what makes zero-copy views possible.
    TensorImpl tensor_impl(storage_2d, {3, 4}, {4, 1});
    EXPECT_EQ(tensor_impl.storage.get(), storage_2d.get());
}

// ════════════════════════════════════════════════════════════════════════════
// TensorImpl — element read access
//
// at({i, j, ...}) computes: offset = sum(index[k] * stride[k])
// and returns storage->ptr()[offset].
//
// We test boundary elements (first, last) and interior elements whose
// expected flat index we can verify by hand.  Using iota storage means
// flat_index == stored_value, so the expected value is just the flat index.
// ════════════════════════════════════════════════════════════════════════════

TEST_F(TensorImplTest, ReadFirstElementOf2DTensor) {
    TensorImpl tensor_impl(storage_2d, {3, 4}, {4, 1});
    // offset = 0*4 + 0*1 = 0 → value = 0.f
    EXPECT_FLOAT_EQ(tensor_impl.at({0, 0}), 0.f);
}

TEST_F(TensorImplTest, ReadLastElementOf2DTensor) {
    TensorImpl tensor_impl(storage_2d, {3, 4}, {4, 1});
    // offset = 2*4 + 3*1 = 11 → value = 11.f
    EXPECT_FLOAT_EQ(tensor_impl.at({2, 3}), 11.f);
}

TEST_F(TensorImplTest, ReadStartOfSecondRowIn2DTensor) {
    // Verifies that the row stride (4) is applied correctly.
    // Moving from row 0 to row 1 must skip exactly 4 elements.
    TensorImpl tensor_impl(storage_2d, {3, 4}, {4, 1});
    // offset = 1*4 + 0*1 = 4 → value = 4.f
    EXPECT_FLOAT_EQ(tensor_impl.at({1, 0}), 4.f);
}

TEST_F(TensorImplTest, ReadInteriorElementOf2DTensor) {
    TensorImpl tensor_impl(storage_2d, {3, 4}, {4, 1});
    // offset = 2*4 + 1*1 = 9 → value = 9.f
    EXPECT_FLOAT_EQ(tensor_impl.at({2, 1}), 9.f);
    // offset = 1*4 + 3*1 = 7 → value = 7.f
    EXPECT_FLOAT_EQ(tensor_impl.at({1, 3}), 7.f);
}

TEST_F(TensorImplTest, ReadFirstElementOf3DTensor) {
    TensorImpl tensor_impl(storage_3d, {2, 3, 4}, {12, 4, 1});
    // offset = 0*12 + 0*4 + 0*1 = 0 → value = 0.f
    EXPECT_FLOAT_EQ(tensor_impl.at({0, 0, 0}), 0.f);
}

TEST_F(TensorImplTest, ReadLastElementOf3DTensor) {
    TensorImpl tensor_impl(storage_3d, {2, 3, 4}, {12, 4, 1});
    // offset = 1*12 + 2*4 + 3*1 = 23 → value = 23.f
    EXPECT_FLOAT_EQ(tensor_impl.at({1, 2, 3}), 23.f);
}

TEST_F(TensorImplTest, ReadStartOfSecondBatchIn3DTensor) {
    // Verifies the outermost stride (12) is applied correctly.
    // Moving from batch 0 to batch 1 must skip 3*4 = 12 elements.
    TensorImpl tensor_impl(storage_3d, {2, 3, 4}, {12, 4, 1});
    // offset = 1*12 + 0*4 + 0*1 = 12 → value = 12.f
    EXPECT_FLOAT_EQ(tensor_impl.at({1, 0, 0}), 12.f);
}

TEST_F(TensorImplTest, ReadInteriorElementOf3DTensor) {
    TensorImpl tensor_impl(storage_3d, {2, 3, 4}, {12, 4, 1});
    // offset = 0*12 + 1*4 + 2*1 = 6 → value = 6.f
    EXPECT_FLOAT_EQ(tensor_impl.at({0, 1, 2}),  6.f);
    // offset = 1*12 + 1*4 + 1*1 = 17 → value = 17.f
    EXPECT_FLOAT_EQ(tensor_impl.at({1, 1, 1}), 17.f);
}

// ════════════════════════════════════════════════════════════════════════════
// TensorImpl — element write access
//
// at() returns a float& so writes go directly into the storage buffer.
// We verify two things separately:
//   1. The write lands at the correct flat index in storage (pointer check).
//   2. A subsequent at() read-back returns the written value (round-trip).
//
// These are separate tests because (1) can catch a wrong-offset write even if
// read uses a different (also wrong) offset that happens to agree with itself.
// ════════════════════════════════════════════════════════════════════════════

TEST_F(TensorImplTest, WriteUpdatesCorrectFlatIndexInStorage2D) {
    TensorImpl tensor_impl(storage_2d, {3, 4}, {4, 1});
    tensor_impl.at({1, 2}) = 99.f;

    // The write must land at flat index 1*4+2 = 6 in the raw storage buffer,
    // not just somewhere that at() happens to read back correctly later.
    EXPECT_FLOAT_EQ(storage_2d->ptr()[6], 99.f);
}

TEST_F(TensorImplTest, WriteReadbackRoundTrip2D) {
    TensorImpl tensor_impl(storage_2d, {3, 4}, {4, 1});
    tensor_impl.at({0, 3}) = 42.f;
    EXPECT_FLOAT_EQ(tensor_impl.at({0, 3}), 42.f);
}

TEST_F(TensorImplTest, WriteUpdatesCorrectFlatIndexInStorage3D) {
    TensorImpl tensor_impl(storage_3d, {2, 3, 4}, {12, 4, 1});
    tensor_impl.at({1, 1, 1}) = 77.f;

    // flat index = 1*12 + 1*4 + 1 = 17
    EXPECT_FLOAT_EQ(storage_3d->ptr()[17], 77.f);
}

TEST_F(TensorImplTest, WriteReadbackRoundTrip3D) {
    TensorImpl tensor_impl(storage_3d, {2, 3, 4}, {12, 4, 1});
    tensor_impl.at({0, 2, 3}) = 55.f;
    EXPECT_FLOAT_EQ(tensor_impl.at({0, 2, 3}), 55.f);
}

TEST_F(TensorImplTest, WriteDoesNotCorruptAdjacentElements) {
    // Writing to one element must not touch its neighbours in flat storage.
    // This catches off-by-one errors in the stride calculation.
    TensorImpl tensor_impl(storage_2d, {3, 4}, {4, 1});
    tensor_impl.at({1, 1}) = 999.f;    // flat index 5

    EXPECT_FLOAT_EQ(tensor_impl.at({1, 0}), 4.f);   // flat index 4 — before
    EXPECT_FLOAT_EQ(tensor_impl.at({1, 2}), 6.f);   // flat index 6 — after
}

// ════════════════════════════════════════════════════════════════════════════
// TensorImpl — shared storage between two instances
//
// Two TensorImpl objects pointing to the same Storage are both live views.
// A write through one must be immediately visible through the other.
// This is the mechanism behind zero-copy transpose and other views —
// if this doesn't work, those operations silently produce stale data.
// ════════════════════════════════════════════════════════════════════════════

TEST_F(TensorImplTest, TwoViewsOverSameStorageSeeEachOthersWrites) {
    TensorImpl view_a(storage_2d, {3, 4}, {4, 1});
    TensorImpl view_b(storage_2d, {3, 4}, {4, 1});

    view_a.at({0, 0}) = 123.f;

    // view_b was not written to directly, but it shares storage_2d,
    // so the value must be visible through it immediately.
    EXPECT_FLOAT_EQ(view_b.at({0, 0}), 123.f);
}

// ════════════════════════════════════════════════════════════════════════════
// TensorImpl — non-standard (transposed) strides
//
// TensorImpl is stride-agnostic: it just computes sum(index[k] * stride[k]).
// Swapping the strides produces a transposed view over the same storage
// without copying any data.
//
// We construct the transposed view manually here (rather than calling
// Tensor::transpose()) to keep this a unit test of TensorImpl's indexing
// logic alone. If we went through Tensor::transpose() and a test failed,
// we couldn't tell whether the bug was in TensorImpl or in transpose().
//
// Original layout [2, 3] with strides {3, 1}, iota data:
//   row-major storage: [0, 1, 2, 3, 4, 5]
//   logical matrix:
//     [0  1  2]
//     [3  4  5]
//
// Transposed view [3, 2] with strides {1, 3}:
//   tT[r, c] = storage[r*1 + c*3]
//   logical matrix:
//     [0  3]
//     [1  4]
//     [2  5]
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorImplTransposedStrides, ReadAccessWithSwappedStrides) {
    std::shared_ptr<Storage> storage = make_iota_storage(6);

    // Transposed view: shape {3, 2}, strides {1, 3} (row and col swapped)
    TensorImpl transposed_view(storage, {3, 2}, {1, 3});

    // First column of tT = first row of original = {0, 1, 2}
    EXPECT_FLOAT_EQ(transposed_view.at({0, 0}), 0.f);
    EXPECT_FLOAT_EQ(transposed_view.at({1, 0}), 1.f);
    EXPECT_FLOAT_EQ(transposed_view.at({2, 0}), 2.f);

    // Second column of tT = second row of original = {3, 4, 5}
    EXPECT_FLOAT_EQ(transposed_view.at({0, 1}), 3.f);
    EXPECT_FLOAT_EQ(transposed_view.at({1, 1}), 4.f);
    EXPECT_FLOAT_EQ(transposed_view.at({2, 1}), 5.f);
}

TEST(TensorImplTransposedStrides, WriteGoesToCorrectFlatIndexViaTransposedStrides) {
    std::shared_ptr<Storage> storage = make_iota_storage(6);
    TensorImpl transposed_view(storage, {3, 2}, {1, 3});

    // tT[1, 0] = storage[1*1 + 0*3] = storage[1]
    transposed_view.at({1, 0}) = 99.f;
    EXPECT_FLOAT_EQ(storage->ptr()[1], 99.f);

    // tT[0, 1] = storage[0*1 + 1*3] = storage[3]
    transposed_view.at({0, 1}) = 77.f;
    EXPECT_FLOAT_EQ(storage->ptr()[3], 77.f);
}

TEST(TensorImplTransposedStrides, WriteThroughTransposedViewVisibleViaOriginalStrides) {
    // If one TensorImpl uses original strides and another uses transposed
    // strides over the same storage, a write through one must be readable
    // through the other at the corresponding logical position.
    std::shared_ptr<Storage> storage = make_iota_storage(6);

    TensorImpl original_view(storage,     {2, 3}, {3, 1});
    TensorImpl transposed_view(storage,   {3, 2}, {1, 3});

    // Write to original[0, 1] — flat index = 0*3 + 1 = 1
    original_view.at({0, 1}) = 55.f;

    // The transposed view of that same element is tT[1, 0] = storage[1*1 + 0*3] = storage[1]
    EXPECT_FLOAT_EQ(transposed_view.at({1, 0}), 55.f);
}