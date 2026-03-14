#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#include "tensor/storage.h"

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

// Returns true if the pointer is aligned to `alignment` bytes.
// Casts to uintptr_t (an integer wide enough to hold a pointer) so we can use
// the modulo operator — you cannot do arithmetic like this on raw pointers.
static bool is_aligned(const void* ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

// ─────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────

class StorageConstructionTest : public ::testing::Test {};

TEST_F(StorageConstructionTest, SizeConstructorDoesNotThrow) {
    EXPECT_NO_THROW(Storage s(1024));
}

TEST_F(StorageConstructionTest, ZeroSizeConstructorDoesNotThrow) {
    EXPECT_NO_THROW(Storage s(0));
}

TEST_F(StorageConstructionTest, DataPointerIsNotNull) {
    Storage s(1024);
    EXPECT_NE(s.data, nullptr);
}

TEST_F(StorageConstructionTest, SizeIsStoredCorrectly) {
    Storage s(1024);
    EXPECT_EQ(s.size, 1024);
}

TEST_F(StorageConstructionTest, LargeAllocationSucceeds) {
    constexpr size_t large = 256 * 1024 * 1024 / sizeof(float);
    EXPECT_NO_THROW(Storage s(large));
}

// ─────────────────────────────────────────────
// Alignment
// ─────────────────────────────────────────────

class StorageAlignmentTest : public ::testing::Test {};

TEST_F(StorageAlignmentTest, DataIs32ByteAligned) {
    Storage s(1024);
    EXPECT_TRUE(is_aligned(s.data, 32));
}

TEST_F(StorageAlignmentTest, SmallSizeIsStill32ByteAligned) {
    Storage s(1);
    EXPECT_TRUE(is_aligned(s.data, 32));
}

TEST_F(StorageAlignmentTest, OddSizeIsStill32ByteAligned) {
    Storage s(7);
    EXPECT_TRUE(is_aligned(s.data, 32));
}

TEST_F(StorageAlignmentTest, MultipleAllocationsAreAllAligned) {
    for (size_t size : {1, 3, 7, 8, 15, 16, 17, 100, 1024, 4096}) {
        Storage s(size);
        EXPECT_TRUE(is_aligned(s.data, 32))
            << "Failed for size " << size;
    }
}

// ─────────────────────────────────────────────
// operator[] — unchecked access
// ─────────────────────────────────────────────

class StorageOperatorTest : public ::testing::Test {};

TEST_F(StorageOperatorTest, WriteThenRead) {
    Storage s(16);
    s[0] = 42.0f;
    EXPECT_FLOAT_EQ(s[0], 42.0f);
}

TEST_F(StorageOperatorTest, AllElementsCanBeWrittenAndRead) {
    constexpr size_t N = 256;
    Storage s(N);
    for (size_t i = 0; i < N; ++i) s[i] = static_cast<float>(i);
    for (size_t i = 0; i < N; ++i)
        EXPECT_FLOAT_EQ(s[i], static_cast<float>(i));
}

TEST_F(StorageOperatorTest, ConstOperatorReturnsCorrectValue) {
    Storage s(4);
    s[0] = 7.0f;
    const Storage& cs = s;
    EXPECT_FLOAT_EQ(cs[0], 7.0f);
}

TEST_F(StorageOperatorTest, WriteToLastElement) {
    constexpr size_t N = 128;
    Storage s(N);
    s[N - 1] = 99.0f;
    EXPECT_FLOAT_EQ(s[N - 1], 99.0f);
}

TEST_F(StorageOperatorTest, ReturnTypeIsReference) {
    // Verifies operator[] returns a reference, not a copy.
    // If it returned a copy, the write through ref would silently do nothing.
    Storage s(4);
    s[0] = 0.0f;
    float& ref = s[0];
    ref = 55.0f;
    EXPECT_FLOAT_EQ(s[0], 55.0f);
}

// ─────────────────────────────────────────────
// at() — bounds-checked access
// ─────────────────────────────────────────────

class StorageAtTest : public ::testing::Test {};

TEST_F(StorageAtTest, ValidIndexDoesNotThrow) {
    Storage s(16);
    EXPECT_NO_THROW(s.at(0));
    EXPECT_NO_THROW(s.at(15));
}

TEST_F(StorageAtTest, WriteThenReadWithAt) {
    Storage s(16);
    s.at(3) = 5.0f;
    EXPECT_FLOAT_EQ(s.at(3), 5.0f);
}

TEST_F(StorageAtTest, OutOfRangeThrowsOutOfRange) {
    Storage s(16);
    EXPECT_THROW(s.at(16), std::out_of_range);
}

TEST_F(StorageAtTest, LargeOutOfRangeIndexThrows) {
    Storage s(16);
    EXPECT_THROW(s.at(9999), std::out_of_range);
}

TEST_F(StorageAtTest, ConstAtValidIndexWorks) {
    Storage s(4);
    s[0] = 3.0f;
    const Storage& cs = s;
    EXPECT_FLOAT_EQ(cs.at(0), 3.0f);
}

TEST_F(StorageAtTest, ConstAtOutOfRangeThrows) {
    Storage s(4);
    const Storage& cs = s;
    EXPECT_THROW(cs.at(4), std::out_of_range);
}

TEST_F(StorageAtTest, AtReturnTypeIsReference) {
    Storage s(4);
    s.at(0) = 0.0f;
    float& ref = s.at(0);
    ref = 77.0f;
    EXPECT_FLOAT_EQ(s.at(0), 77.0f);
}

TEST_F(StorageAtTest, OperatorAndAtAgreeOnValues) {
    // Both accessors should read from the same underlying memory.
    Storage s(8);
    for (size_t i = 0; i < 8; ++i) s[i] = static_cast<float>(i * 10);
    for (size_t i = 0; i < 8; ++i)
        EXPECT_FLOAT_EQ(s[i], s.at(i));
}

// ─────────────────────────────────────────────
// fill()
// ─────────────────────────────────────────────

class StorageFillTest : public ::testing::Test {};

TEST_F(StorageFillTest, FillSetsAllElements) {
    Storage s(64);
    s.fill(1.0f);
    for (size_t i = 0; i < s.size; ++i)
        EXPECT_FLOAT_EQ(s[i], 1.0f) << "Mismatch at index " << i;
}

TEST_F(StorageFillTest, FillWithZero) {
    // Underpins Tensor::zeros — worth testing explicitly.
    Storage s(64);
    s.fill(1.0f); // dirty first
    s.fill(0.0f);
    for (size_t i = 0; i < s.size; ++i)
        EXPECT_FLOAT_EQ(s[i], 0.0f) << "Mismatch at index " << i;
}

TEST_F(StorageFillTest, FillWithOne) {
    // Underpins Tensor::ones.
    Storage s(64);
    s.fill(1.0f);
    for (size_t i = 0; i < s.size; ++i)
        EXPECT_FLOAT_EQ(s[i], 1.0f) << "Mismatch at index " << i;
}

TEST_F(StorageFillTest, FillOverwritesPreviousValues) {
    Storage s(16);
    for (size_t i = 0; i < s.size; ++i) s[i] = static_cast<float>(i);
    s.fill(7.0f);
    for (size_t i = 0; i < s.size; ++i)
        EXPECT_FLOAT_EQ(s[i], 7.0f) << "Mismatch at index " << i;
}

TEST_F(StorageFillTest, FillWithNegativeValue) {
    Storage s(16);
    s.fill(-3.5f);
    for (size_t i = 0; i < s.size; ++i)
        EXPECT_FLOAT_EQ(s[i], -3.5f) << "Mismatch at index " << i;
}

TEST_F(StorageFillTest, FillWithLargeStorage) {
    constexpr size_t N = 1024 * 1024;
    Storage s(N);
    s.fill(2.0f);
    // Spot-check rather than iterating 1M elements in a test.
    EXPECT_FLOAT_EQ(s[0], 2.0f);
    EXPECT_FLOAT_EQ(s[N / 2], 2.0f);
    EXPECT_FLOAT_EQ(s[N - 1], 2.0f);
}

TEST_F(StorageFillTest, FillCanBeCalledMultipleTimes) {
    Storage s(16);
    s.fill(1.0f);
    s.fill(2.0f);
    s.fill(3.0f);
    for (size_t i = 0; i < s.size; ++i)
        EXPECT_FLOAT_EQ(s[i], 3.0f) << "Mismatch at index " << i;
}

// ─────────────────────────────────────────────
// Copy semantics (must be disabled)
// ─────────────────────────────────────────────

class StorageCopyTest : public ::testing::Test {};

TEST_F(StorageCopyTest, CopyConstructorIsDeleted) {
    EXPECT_FALSE(std::is_copy_constructible<Storage>::value);
}

TEST_F(StorageCopyTest, CopyAssignmentIsDeleted) {
    EXPECT_FALSE(std::is_copy_assignable<Storage>::value);
}

// ─────────────────────────────────────────────
// Move semantics
// ─────────────────────────────────────────────

class StorageMoveTest : public ::testing::Test {};

TEST_F(StorageMoveTest, MoveConstructorIsNoexcept) {
    EXPECT_TRUE(std::is_nothrow_move_constructible<Storage>::value);
}

TEST_F(StorageMoveTest, MoveAssignmentIsNoexcept) {
    EXPECT_TRUE(std::is_nothrow_move_assignable<Storage>::value);
}

TEST_F(StorageMoveTest, MoveConstructorTransfersPointer) {
    Storage a(64);
    float* original_ptr = a.data;
    Storage b(std::move(a));
    EXPECT_EQ(b.data, original_ptr);
}

TEST_F(StorageMoveTest, MoveConstructorNullsSourcePointer) {
    Storage a(64);
    Storage b(std::move(a));
    EXPECT_EQ(a.data, nullptr);
}

TEST_F(StorageMoveTest, MoveConstructorTransfersSize) {
    Storage a(64);
    Storage b(std::move(a));
    EXPECT_EQ(b.size, 64);
    EXPECT_EQ(a.size, 0);
}

TEST_F(StorageMoveTest, MoveAssignmentTransfersPointer) {
    Storage a(64);
    Storage b(32);
    float* original_ptr = a.data;
    b = std::move(a);
    EXPECT_EQ(b.data, original_ptr);
    EXPECT_EQ(a.data, nullptr);
}

TEST_F(StorageMoveTest, MoveAssignmentTransfersSize) {
    Storage a(64);
    Storage b(32);
    b = std::move(a);
    EXPECT_EQ(b.size, 64);
    EXPECT_EQ(a.size, 0);
}

TEST_F(StorageMoveTest, MovedFromObjectCanBeDestroyedSafely) {
    Storage a(64);
    {
        Storage b(std::move(a));
    }
    // a's destructor runs here with data == nullptr — must not crash.
}

TEST_F(StorageMoveTest, SelfMoveAssignmentIsSafe) {
    Storage a(64);
    float* ptr = a.data;
    a = std::move(a);
    EXPECT_EQ(a.data, ptr);
}

TEST_F(StorageMoveTest, DataIsPreservedAfterMove) {
    Storage a(4);
    a[0] = 1.0f;
    a[1] = 2.0f;
    Storage b(std::move(a));
    EXPECT_FLOAT_EQ(b[0], 1.0f);
    EXPECT_FLOAT_EQ(b[1], 2.0f);
}

// ─────────────────────────────────────────────
// Shared ownership via shared_ptr
// ─────────────────────────────────────────────

class StorageSharedPtrTest : public ::testing::Test {};

TEST_F(StorageSharedPtrTest, SharedPtrCanOwnStorage) {
    auto s = std::make_shared<Storage>(64);
    EXPECT_NE(s->data, nullptr);
}

TEST_F(StorageSharedPtrTest, TwoSharedPtrsShareSameData) {
    auto a = std::make_shared<Storage>(64);
    std::shared_ptr<Storage> b = a;
    EXPECT_EQ(a->data, b->data);
}

TEST_F(StorageSharedPtrTest, OperatorWorksViaSharedPtr) {
    auto s = std::make_shared<Storage>(16);
    (*s)[0] = 42.0f;
    EXPECT_FLOAT_EQ((*s)[0], 42.0f);
}

TEST_F(StorageSharedPtrTest, AtWorksViaSharedPtr) {
    auto s = std::make_shared<Storage>(16);
    s->at(0) = 42.0f;
    EXPECT_FLOAT_EQ(s->at(0), 42.0f);
}

TEST_F(StorageSharedPtrTest, DataSurvivesUntilLastOwnerDies) {
    float* raw = nullptr;
    {
        auto a = std::make_shared<Storage>(64);
        raw = a->data;
        {
            std::shared_ptr<Storage> b = a;
        }
        // b died, but a still holds it
        EXPECT_EQ(a->data, raw);
    }
}

TEST_F(StorageSharedPtrTest, RefcountReflectsOwners) {
    auto a = std::make_shared<Storage>(64);
    EXPECT_EQ(a.use_count(), 1);
    {
        auto b = a;
        EXPECT_EQ(a.use_count(), 2);
    }
    EXPECT_EQ(a.use_count(), 1);
}