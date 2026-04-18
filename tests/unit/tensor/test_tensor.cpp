#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor/tensor.h"
#include "autograd.h"

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

// Builds a Shape array from an initializer list and returns the element count
// as ndim. Using a helper avoids repeating Shape construction in every test.
static Shape make_shape(std::initializer_list<size_t> dims) {
    assert(dims.size() <= MAX_DIM);
    Shape s{};
    size_t i = 0;
    for (size_t d : dims) s[i++] = d;
    return s;
}

static Tensor make_tensor(std::initializer_list<size_t> dims) {
    return Tensor(make_shape(dims), dims.size());
}

static Tensor make_zeros(std::initializer_list<size_t> dims) {
    return Tensor::zeros(make_shape(dims), dims.size());
}

static Tensor make_ones(std::initializer_list<size_t> dims) {
    return Tensor::ones(make_shape(dims), dims.size());
}

// ─────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────

class TensorConstructionTest : public ::testing::Test {};

TEST_F(TensorConstructionTest, BasicConstructionDoesNotThrow) {
    EXPECT_NO_THROW(make_tensor({4, 4}));
}

TEST_F(TensorConstructionTest, OneDimensional) {
    EXPECT_NO_THROW(make_tensor({16}));
}

TEST_F(TensorConstructionTest, HighDimensional) {
    // MAX_DIM is 8 — this should be the maximum valid rank.
    EXPECT_NO_THROW(make_tensor({2, 2, 2, 2, 2, 2, 2, 2}));
}

TEST_F(TensorConstructionTest, ExceedingMaxDimAsserts) {
    // Shape is fixed at MAX_DIM slots. Passing ndim > MAX_DIM with a
    // full Shape should trigger the assert inside the constructor.
    Shape shape{};
    shape.fill(2);
    EXPECT_DEATH(Tensor(shape, MAX_DIM + 1), "");
}

// Note: NullShapeAsserts is no longer applicable — Shape is a value type,
// not a pointer, so it can never be null.

// ─────────────────────────────────────────────
// zeros and ones
// ─────────────────────────────────────────────

class TensorFactoryTest : public ::testing::Test {};

TEST_F(TensorFactoryTest, ZerosAreAllZero) {
    auto t = make_zeros({4, 4});
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j) {
            float val = t(i, j);
            EXPECT_FLOAT_EQ(val, 0.0f) << "Failed at [" << i << "," << j << "]";
        }
}

TEST_F(TensorFactoryTest, OnesAreAllOne) {
    auto t = make_ones({4, 4});
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j) {
            float val = t(i, j);
            EXPECT_FLOAT_EQ(val, 1.0f) << "Failed at [" << i << "," << j << "]";
        }
}

TEST_F(TensorFactoryTest, ZerosOneDimensional) {
    auto t = make_zeros({16});
    for (size_t i = 0; i < 16; ++i)
        EXPECT_FLOAT_EQ(t(i), 0.0f);
}

TEST_F(TensorFactoryTest, OnesOneDimensional) {
    auto t = make_ones({16});
    for (size_t i = 0; i < 16; ++i)
        EXPECT_FLOAT_EQ(t(i), 1.0f);
}

TEST_F(TensorFactoryTest, ZerosAndOnesAreIndependent) {
    auto a = make_zeros({4});
    auto b = make_ones({4});
    EXPECT_FLOAT_EQ(a(0), 0.0f);
    EXPECT_FLOAT_EQ(b(0), 1.0f);
}

// ─────────────────────────────────────────────
// Strides
// ─────────────────────────────────────────────

class TensorStrideTest : public ::testing::Test {};

TEST_F(TensorStrideTest, OneDimensionalStrideIsOne) {
    auto t = make_tensor({16});
    EXPECT_EQ(t.strides[0], 1);
}

TEST_F(TensorStrideTest, TwoDimensionalRowMajorStrides) {
    // Shape {4, 8}: last dim stride = 1, first dim stride = 8.
    auto t = make_tensor({4, 8});
    EXPECT_EQ(t.strides[1], 1);
    EXPECT_EQ(t.strides[0], 8);
}

TEST_F(TensorStrideTest, ThreeDimensionalRowMajorStrides) {
    // Shape {2, 4, 8}: strides should be {32, 8, 1}.
    auto t = make_tensor({2, 4, 8});
    EXPECT_EQ(t.strides[2], 1);
    EXPECT_EQ(t.strides[1], 8);
    EXPECT_EQ(t.strides[0], 32);
}

TEST_F(TensorStrideTest, FourDimensionalRowMajorStrides) {
    // Shape {2, 3, 4, 5}: strides should be {60, 20, 5, 1}.
    auto t = make_tensor({2, 3, 4, 5});
    EXPECT_EQ(t.strides[3], 1);
    EXPECT_EQ(t.strides[2], 5);
    EXPECT_EQ(t.strides[1], 20);
    EXPECT_EQ(t.strides[0], 60);
}

TEST_F(TensorStrideTest, StridesFromShapeMatchManual) {
    // Verify the static helper directly.
    Shape shape = make_shape({3, 4, 5});
    Strides s = Tensor::strides_from_shape(shape, 3);
    EXPECT_EQ(s[2], 1);
    EXPECT_EQ(s[1], 5);
    EXPECT_EQ(s[0], 20);
}

// ─────────────────────────────────────────────
// Indexing — operator()
// ─────────────────────────────────────────────

class TensorIndexTest : public ::testing::Test {};

TEST_F(TensorIndexTest, WriteThenRead1D) {
    auto t = make_tensor({16});
    t(5) = 42.0f;
    EXPECT_FLOAT_EQ(t(5), 42.0f);
}

TEST_F(TensorIndexTest, WriteThenRead2D) {
    auto t = make_tensor({4, 4});
    t(2, 3) = 7.0f;
    float val = t(2, 3);
    EXPECT_FLOAT_EQ(val, 7.0f);
}

TEST_F(TensorIndexTest, WriteThenRead3D) {
    auto t = make_tensor({2, 4, 8});
    t(1, 2, 3) = 9.0f;
    float val = t(1, 2, 3);
    EXPECT_FLOAT_EQ(val, 9.0f);
}

TEST_F(TensorIndexTest, IndependentElementsDoNotAlias) {
    auto t = make_zeros({4, 4});
    t(1, 1) = 5.0f;
    EXPECT_FLOAT_EQ(t(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(t(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(t(2, 2), 0.0f);
}

TEST_F(TensorIndexTest, FlatLayoutIsRowMajor) {
    // Fill underlying storage with sequential values, then verify
    // that (i, j) maps to flat position i*cols + j.
    constexpr size_t rows = 3, cols = 4;
    auto t = make_tensor({rows, cols});
    for (size_t k = 0; k < rows * cols; ++k)
        (*t.storage)[k] = static_cast<float>(k);

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) {
            float val = t(i, j);
            EXPECT_FLOAT_EQ(val, static_cast<float>(i * cols + j))
                << "Failed at (" << i << "," << j << ")";
        }
}

TEST_F(TensorIndexTest, AllElementsWritable) {
    constexpr size_t N = 8;
    auto t = make_tensor({N, N});
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            t(i, j) = static_cast<float>(i * N + j);

    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) {
            float val = t(i, j);
            EXPECT_FLOAT_EQ(val, static_cast<float>(i * N + j));
        }
}

TEST_F(TensorIndexTest, WrongNumberOfIndicesAsserts) {
    auto t = make_tensor({4, 4});
    EXPECT_DEATH(t(0), "");
}

TEST_F(TensorIndexTest, NonIntegerIndexFailsCompilation) {
    // static_assert in operator() rejects non-integer indices at compile time.
    // We document the intent here by testing the underlying trait directly.
    EXPECT_TRUE((std::is_integral_v<int>));
    EXPECT_FALSE((std::is_integral_v<float>));
}

// ─────────────────────────────────────────────
// numel
// ─────────────────────────────────────────────

class TensorNumelTest : public ::testing::Test {};

TEST_F(TensorNumelTest, OneDimensional) {
    auto t = make_tensor({16});
    EXPECT_EQ(t.numel, 16);
}

TEST_F(TensorNumelTest, TwoDimensional) {
    auto t = make_tensor({4, 8});
    EXPECT_EQ(t.numel, 32);
}

TEST_F(TensorNumelTest, ThreeDimensional) {
    auto t = make_tensor({2, 3, 4});
    EXPECT_EQ(t.numel, 24);
}

TEST_F(TensorNumelTest, HighDimensional) {
    auto t = make_tensor({2, 2, 2, 2, 2, 2, 2, 2});
    EXPECT_EQ(t.numel, 256);
}

// ─────────────────────────────────────────────
// Shared storage
// ─────────────────────────────────────────────

class TensorStorageTest : public ::testing::Test {};

TEST_F(TensorStorageTest, StorageIsNotNull) {
    auto t = make_tensor({4, 4});
    EXPECT_NE(t.storage, nullptr);
}

TEST_F(TensorStorageTest, StorageSizeMatchesNumel) {
    auto t = make_tensor({4, 8});
    EXPECT_EQ(t.storage->size, t.numel);
}

TEST_F(TensorStorageTest, TwoTensorsHaveIndependentStorage) {
    auto a = make_ones({4});
    auto b = make_zeros({4});
    EXPECT_NE(a.storage, b.storage);
}

TEST_F(TensorStorageTest, SharedStorageReflectsWrites) {
    // Two tensors sharing the same storage should see each other's writes.
    auto a = make_zeros({4});
    Tensor b = a; // shared_ptr copy — same underlying storage
    a(2) = 99.0f;
    EXPECT_FLOAT_EQ(b(2), 99.0f);
}

// ─────────────────────────────────────────────
// Per-dim accessors: shape_at / stride_at
// ─────────────────────────────────────────────

class TensorDimAccessorTest : public ::testing::Test {};

TEST_F(TensorDimAccessorTest, ShapeAtReturnsCorrectDims) {
    auto t = make_tensor({2, 3, 4});
    EXPECT_EQ(t.shape_at(0), 2u);
    EXPECT_EQ(t.shape_at(1), 3u);
    EXPECT_EQ(t.shape_at(2), 4u);
}

TEST_F(TensorDimAccessorTest, StrideAtReturnsCorrectDims) {
    // shape {2, 3, 4} → strides {12, 4, 1}
    auto t = make_tensor({2, 3, 4});
    EXPECT_EQ(t.stride_at(0), 12u);
    EXPECT_EQ(t.stride_at(1),  4u);
    EXPECT_EQ(t.stride_at(2),  1u);
}

TEST_F(TensorDimAccessorTest, ShapeAtOneDim) {
    auto t = make_tensor({16});
    EXPECT_EQ(t.shape_at(0), 16u);
}

TEST_F(TensorDimAccessorTest, StrideAtOneDim) {
    auto t = make_tensor({16});
    EXPECT_EQ(t.stride_at(0), 1u);
}

TEST_F(TensorDimAccessorTest, ShapeAtOutOfRangeThrows) {
    auto t = make_tensor({4, 4});
    EXPECT_THROW(t.shape_at(2),  std::out_of_range);
    EXPECT_THROW(t.shape_at(99), std::out_of_range);
}

TEST_F(TensorDimAccessorTest, StrideAtOutOfRangeThrows) {
    auto t = make_tensor({4, 4});
    EXPECT_THROW(t.stride_at(2),  std::out_of_range);
    EXPECT_THROW(t.stride_at(99), std::out_of_range);
}

TEST_F(TensorDimAccessorTest, ShapeAtLastValidDimDoesNotThrow) {
    auto t = make_tensor({2, 3, 4, 5});
    EXPECT_NO_THROW(t.shape_at(3));
    EXPECT_EQ(t.shape_at(3), 5u);
}

// ─────────────────────────────────────────────
// clone()
// ─────────────────────────────────────────────

class TensorCloneTest : public ::testing::Test {};

TEST_F(TensorCloneTest, CloneHasSameShape) {
    auto t = make_tensor({3, 4});
    auto c = t.clone();
    EXPECT_EQ(c.ndim, t.ndim);
    EXPECT_EQ(c.shape_at(0), t.shape_at(0));
    EXPECT_EQ(c.shape_at(1), t.shape_at(1));
}

TEST_F(TensorCloneTest, CloneHasSameValues) {
    auto t = make_zeros({3, 4});
    t(1, 2) = 7.0f;
    t(2, 3) = 5.0f;
    auto c = t.clone();
    EXPECT_FLOAT_EQ(c(1, 2), 7.0f);
    EXPECT_FLOAT_EQ(c(2, 3), 5.0f);
}

TEST_F(TensorCloneTest, CloneHasIndependentStorage) {
    auto t = make_zeros({4});
    auto c = t.clone();
    t(0) = 99.0f;
    EXPECT_FLOAT_EQ(c(0), 0.0f);  // clone not affected by write to original
}

TEST_F(TensorCloneTest, WriteToCloneDoesNotAffectOriginal) {
    auto t = make_ones({4});
    auto c = t.clone();
    c(0) = 42.0f;
    EXPECT_FLOAT_EQ(t(0), 1.0f);  // original unchanged
}

TEST_F(TensorCloneTest, CloneIsContiguous) {
    auto t = make_tensor({3, 4});
    auto c = t.clone();
    EXPECT_TRUE(c.is_contiguous());
}

TEST_F(TensorCloneTest, CloneHasNoAutogradMeta) {
    // Even if the source has requires_grad, clone produces a detached copy.
    auto s = make_shape({4});
    auto t = Tensor::zeros(s, 1, /*requires_grad=*/true);
    auto c = t.clone();
    EXPECT_FALSE(c.requires_grad());
}

TEST_F(TensorCloneTest, CloneOf1DOnesTones) {
    auto t = make_ones({8});
    auto c = t.clone();
    for (size_t i = 0; i < 8; ++i)
        EXPECT_FLOAT_EQ(c(i), 1.0f);
}

TEST_F(TensorCloneTest, ClonePreservesNumel) {
    auto t = make_tensor({2, 3, 4});
    auto c = t.clone();
    EXPECT_EQ(c.numel, t.numel);
}

// ─────────────────────────────────────────────
// requires_grad and autograd_meta
// ─────────────────────────────────────────────

class TensorAutogradMetaTest : public ::testing::Test {};

TEST_F(TensorAutogradMetaTest, DefaultTensorHasNoAutogradMeta) {
    auto t = make_tensor({4, 4});
    EXPECT_EQ(t.autograd_meta, nullptr);
}

TEST_F(TensorAutogradMetaTest, DefaultRequiresGradIsFalse) {
    auto t = make_tensor({4, 4});
    EXPECT_FALSE(t.requires_grad());
}

TEST_F(TensorAutogradMetaTest, ZerosWithRequiresGradTrue) {
    auto s = make_shape({3, 3});
    auto t = Tensor::zeros(s, 2, /*requires_grad=*/true);
    EXPECT_TRUE(t.requires_grad());
}

TEST_F(TensorAutogradMetaTest, OnesWithRequiresGradTrue) {
    auto s = make_shape({3});
    auto t = Tensor::ones(s, 1, /*requires_grad=*/true);
    EXPECT_TRUE(t.requires_grad());
}

TEST_F(TensorAutogradMetaTest, ZerosWithRequiresGradFalseDefault) {
    auto s = make_shape({3, 3});
    auto t = Tensor::zeros(s, 2);
    EXPECT_FALSE(t.requires_grad());
    EXPECT_EQ(t.autograd_meta, nullptr);
}

TEST_F(TensorAutogradMetaTest, RequiresGradTensorHasNonNullMeta) {
    auto s = make_shape({4});
    auto t = Tensor::zeros(s, 1, /*requires_grad=*/true);
    EXPECT_NE(t.autograd_meta, nullptr);
}

TEST_F(TensorAutogradMetaTest, HasGradFalseBeforeBackward) {
    auto s = make_shape({4});
    auto t = Tensor::zeros(s, 1, /*requires_grad=*/true);
    EXPECT_FALSE(t.has_grad());
}

TEST_F(TensorAutogradMetaTest, GradThrowsWhenNoGradient) {
    auto s = make_shape({4});
    auto t = Tensor::zeros(s, 1, /*requires_grad=*/true);
    EXPECT_THROW(t.grad(), std::runtime_error);
}

TEST_F(TensorAutogradMetaTest, GradThrowsOnPlainTensor) {
    auto t = make_zeros({4});
    EXPECT_THROW(t.grad(), std::runtime_error);
}

TEST_F(TensorAutogradMetaTest, ManuallySetGradIsRetrievable) {
    auto s = make_shape({2});
    auto t = Tensor::ones(s, 1, /*requires_grad=*/true);

    // Simulate what backward() would do: populate the grad field.
    auto g = std::make_shared<Tensor>(Tensor::zeros(s, 1));
    (*g)(0) = 3.0f;
    (*g)(1) = 7.0f;
    t.autograd_meta->grad = g;

    EXPECT_TRUE(t.has_grad());
    EXPECT_FLOAT_EQ(t.grad()(0), 3.0f);
    EXPECT_FLOAT_EQ(t.grad()(1), 7.0f);
}

// ─────────────────────────────────────────────
// from_storage
// ─────────────────────────────────────────────

class TensorFromStorageTest : public ::testing::Test {};

TEST_F(TensorFromStorageTest, WrapsExistingStorage) {
    auto s = make_shape({3});
    auto stor = std::make_shared<Storage>(3);
    stor->data[0] = 1.0f;
    stor->data[1] = 2.0f;
    stor->data[2] = 3.0f;

    auto t = Tensor::from_storage(stor, s, 1);
    EXPECT_FLOAT_EQ(t(0), 1.0f);
    EXPECT_FLOAT_EQ(t(1), 2.0f);
    EXPECT_FLOAT_EQ(t(2), 3.0f);
}

TEST_F(TensorFromStorageTest, SharesOwnership) {
    auto s    = make_shape({2});
    auto stor = std::make_shared<Storage>(2);
    stor->data[0] = 5.0f;
    stor->data[1] = 6.0f;

    auto t = Tensor::from_storage(stor, s, 1);
    // Writing through the tensor should be visible via the original shared_ptr.
    t(0) = 99.0f;
    EXPECT_FLOAT_EQ(stor->data[0], 99.0f);
}

TEST_F(TensorFromStorageTest, ShapeAndNumelAreCorrect) {
    auto s    = make_shape({2, 3});
    auto stor = std::make_shared<Storage>(6);
    auto t    = Tensor::from_storage(stor, s, 2);
    EXPECT_EQ(t.numel, 6u);
    EXPECT_EQ(t.shape_at(0), 2u);
    EXPECT_EQ(t.shape_at(1), 3u);
}