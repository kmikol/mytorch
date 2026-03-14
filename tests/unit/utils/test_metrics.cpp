#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "utils/metrics.h"

class MetricsTest : public ::testing::Test {};

TEST_F(MetricsTest, AccuracyIsOneWhenAllPredictionsMatch) {
    const std::vector<size_t> pred = {0, 1, 2, 3};
    const std::vector<size_t> gt = {0, 1, 2, 3};

    const Metrics m = compute_metrics(pred, gt);
    EXPECT_FLOAT_EQ(m.accuracy, 1.0f);
}

TEST_F(MetricsTest, AccuracyIsZeroWhenNoPredictionsMatch) {
    const std::vector<size_t> pred = {0, 1, 2, 3};
    const std::vector<size_t> gt = {4, 5, 6, 7};

    const Metrics m = compute_metrics(pred, gt);
    EXPECT_FLOAT_EQ(m.accuracy, 0.0f);
}

TEST_F(MetricsTest, AccuracyIsFractionForPartialMatches) {
    const std::vector<size_t> pred = {1, 0, 2, 7, 4};
    const std::vector<size_t> gt = {1, 9, 2, 3, 4};

    const Metrics m = compute_metrics(pred, gt);
    EXPECT_FLOAT_EQ(m.accuracy, 3.0f / 5.0f);
}

TEST_F(MetricsTest, EmptyInputsReturnZeroAccuracy) {
    const std::vector<size_t> pred = {};
    const std::vector<size_t> gt = {};

    const Metrics m = compute_metrics(pred, gt);
    EXPECT_FLOAT_EQ(m.accuracy, 0.0f);
}

TEST_F(MetricsTest, SizeMismatchAsserts) {
    const std::vector<size_t> pred = {0, 1, 2};
    const std::vector<size_t> gt = {0, 1};

    EXPECT_DEATH(compute_metrics(pred, gt), "");
}

TEST_F(MetricsTest, LargeDatasetWithDeterministicNoiseMatchesReference) {
    constexpr size_t n = 10000;
    std::vector<size_t> pred(n);
    std::vector<size_t> gt(n);

    // Ground truth cycles over 10 classes. Predictions are correct except
    // every 7th sample where we force a different class.
    for (size_t i = 0; i < n; ++i) {
        gt[i] = i % 10;
        pred[i] = gt[i];
        if (i % 7 == 0)
            pred[i] = (gt[i] + 1) % 10;
    }

    size_t correct = 0;
    for (size_t i = 0; i < n; ++i)
        correct += (pred[i] == gt[i]) ? 1u : 0u;

    const float expected = static_cast<float>(correct) / static_cast<float>(n);
    const Metrics m = compute_metrics(pred, gt);

    EXPECT_FLOAT_EQ(m.accuracy, expected);
}

TEST_F(MetricsTest, HighlyImbalancedLabelsStillComputeExactAccuracy) {
    // 950 zeros + 50 ones; predict all zeros.
    std::vector<size_t> gt(1000, 0u);
    for (size_t i = 950; i < 1000; ++i)
        gt[i] = 1u;

    std::vector<size_t> pred(1000, 0u);

    const Metrics m = compute_metrics(pred, gt);
    EXPECT_FLOAT_EQ(m.accuracy, 0.95f);
}

TEST_F(MetricsTest, SameLabelHistogramDifferentOrderIsNotPerfectAccuracy) {
    // pred has the same histogram as gt but shifted by one position;
    // this validates position-wise comparison rather than set comparison.
    const std::vector<size_t> gt = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<size_t> pred = {9, 0, 1, 2, 3, 4, 5, 6, 7, 8};

    const Metrics m = compute_metrics(pred, gt);
    EXPECT_FLOAT_EQ(m.accuracy, 0.0f);
}

TEST_F(MetricsTest, AccuracyStaysWithinClosedUnitInterval) {
    std::vector<size_t> gt;
    std::vector<size_t> pred;
    gt.reserve(2048);
    pred.reserve(2048);

    for (size_t i = 0; i < 2048; ++i) {
        gt.push_back((i * 17 + 3) % 10);
        pred.push_back((i * 29 + 1) % 10);
    }

    const Metrics m = compute_metrics(pred, gt);
    EXPECT_GE(m.accuracy, 0.0f);
    EXPECT_LE(m.accuracy, 1.0f);
}
