#include <gtest/gtest.h>
#include "tensor/storage.h"   // adjust path if needed

// ── fixture: shared setup for tests that need a pre-filled storage ──

class StorageTest : public ::testing::Test {
protected:
    void SetUp() override {
        filled = Storage(6, 0.f);
    }
    Storage filled;   // 6 elements, all zero
};

// ── default constructor ──────────────────────────────────────────────

TEST(StorageDefault, StartsEmpty) {
    Storage s;
    EXPECT_EQ(s.data.size(), 0u);
}

TEST(StorageDefault, PtrOnEmptyIsNullOrSize0) {
    Storage s;
    // either ptr() returns nullptr or data is simply empty —
    // both are valid; we just confirm no elements exist
    EXPECT_EQ(s.data.size(), 0u);
}

// ── sized constructor ────────────────────────────────────────────────

TEST_F(StorageTest, SizeIsCorrect) {
    EXPECT_EQ(filled.data.size(), 6u);
}

TEST_F(StorageTest, PtrIsNonNull) {
    EXPECT_NE(filled.ptr(), nullptr);
}

TEST_F(StorageTest, FilledWithRequestedValue) {
    Storage s(4, 3.14f);
    for (int i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(s.ptr()[i], 3.14f) << "mismatch at index " << i;
}

TEST_F(StorageTest, FilledWithZero) {
    for (int i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(filled.ptr()[i], 0.f) << "mismatch at index " << i;
}

// ── read / write via ptr() ───────────────────────────────────────────

TEST_F(StorageTest, WriteAndReadBack) {
    filled.ptr()[2] = 3.14f;
    EXPECT_FLOAT_EQ(filled.ptr()[2], 3.14f);
}

TEST_F(StorageTest, WritingOneIndexLeavesOthersUntouched) {
    filled.ptr()[3] = 99.f;
    EXPECT_FLOAT_EQ(filled.ptr()[0], 0.f);
    EXPECT_FLOAT_EQ(filled.ptr()[5], 0.f);
}

// ── independence: two Storage objects don't share memory ────────────

TEST(StorageIndependence, MutatingADoesNotAffectB) {
    Storage a(4, 1.f);
    Storage b(4, 2.f);

    a.ptr()[0] = 99.f;

    EXPECT_FLOAT_EQ(b.ptr()[0], 2.f);   // b is untouched
}

TEST(StorageIndependence, PtrsAreDifferent) {
    Storage a(4, 0.f);
    Storage b(4, 0.f);
    EXPECT_NE(a.ptr(), b.ptr());
}

// ── edge cases ───────────────────────────────────────────────────────

TEST(StorageEdge, SizeOneWorks) {
    Storage s(1, 42.f);
    EXPECT_EQ(s.data.size(), 1u);
    EXPECT_FLOAT_EQ(s.ptr()[0], 42.f);
}

TEST(StorageEdge, LargeAllocation) {
    Storage s(1024 * 1024, 0.f);
    EXPECT_EQ(s.data.size(), 1024u * 1024u);
    s.ptr()[1024 * 1024 - 1] = 7.f;
    EXPECT_FLOAT_EQ(s.ptr()[1024 * 1024 - 1], 7.f);
}