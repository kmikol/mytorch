#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <vector>

#include "loss_functions/cross_entropy.h"

namespace {

static Shape make_shape(std::initializer_list<size_t> dims) {
    Shape s{};
    size_t i = 0;
    for (size_t d : dims)
        s[i++] = d;
    return s;
}

static Tensor make_tensor(std::initializer_list<size_t> dims,
                          std::initializer_list<float> vals,
                          bool requires_grad = false) {
    Shape s = make_shape(dims);
    const size_t ndim = dims.size();
    Tensor t(s, ndim, requires_grad);

    assert(vals.size() == t.numel);

    size_t k = 0;
    const size_t cols = (ndim == 2) ? s[1] : 0;
    for (float v : vals) {
        if (ndim == 2)
            t(k / cols, k % cols) = v;
        else
            t(k) = v;
        ++k;
    }
    return t;
}

static std::vector<float> softmax_row(const Tensor& logits, size_t row) {
    const size_t C = logits.shape_at(1);
    float row_max = logits(row, 0);
    for (size_t j = 1; j < C; ++j)
        row_max = std::max(row_max, logits(row, j));

    std::vector<float> out(C, 0.f);
    float sum = 0.f;
    for (size_t j = 0; j < C; ++j) {
        out[j] = std::exp(logits(row, j) - row_max);
        sum += out[j];
    }

    for (size_t j = 0; j < C; ++j)
        out[j] /= sum;

    return out;
}

}  // namespace

class CrossEntropyOpForwardTest : public ::testing::Test {};

TEST_F(CrossEntropyOpForwardTest, OutputIsScalar) {
    auto logits  = make_tensor({2, 3}, {2.f, 1.f, 0.f, 0.f, 2.f, 1.f});
    auto targets = make_tensor({2, 3}, {1.f, 0.f, 0.f, 0.f, 1.f, 0.f});
    auto loss = CrossEntropyOp::forward(logits, targets);

    EXPECT_EQ(loss.ndim, 1u);
    EXPECT_EQ(loss.shape_at(0), 1u);
    EXPECT_EQ(loss.numel, 1u);
}

TEST_F(CrossEntropyOpForwardTest, UniformLogitsGiveLogClassCount) {
    auto logits  = make_tensor({1, 4}, {0.f, 0.f, 0.f, 0.f});
    auto targets = make_tensor({1, 4}, {1.f, 0.f, 0.f, 0.f});
    auto loss = CrossEntropyOp::forward(logits, targets);

    EXPECT_NEAR(loss(0), std::log(4.f), 1e-6f);
}

TEST_F(CrossEntropyOpForwardTest, PerfectLargeMarginPredictionIsNearZero) {
    auto logits  = make_tensor({1, 3}, {12.f, -6.f, -8.f});
    auto targets = make_tensor({1, 3}, {1.f, 0.f, 0.f});
    auto loss = CrossEntropyOp::forward(logits, targets);

    EXPECT_LT(loss(0), 1e-6f);
}

TEST_F(CrossEntropyOpForwardTest, ShiftInvariancePerRow) {
    auto logits_a = make_tensor({2, 3}, {1.f, 3.f, 2.f, 5.f, 0.f, -1.f});
    auto logits_b = make_tensor({2, 3}, {11.f, 13.f, 12.f, 1.f, -4.f, -5.f});
    auto targets  = make_tensor({2, 3}, {0.f, 1.f, 0.f, 1.f, 0.f, 0.f});

    auto loss_a = CrossEntropyOp::forward(logits_a, targets);
    auto loss_b = CrossEntropyOp::forward(logits_b, targets);

    EXPECT_NEAR(loss_a(0), loss_b(0), 1e-6f);
}

TEST_F(CrossEntropyOpForwardTest, ShapeMismatchAsserts) {
    auto logits  = make_tensor({2, 3}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
    auto targets = make_tensor({2, 2}, {1.f, 0.f, 0.f, 1.f});

    EXPECT_DEATH(CrossEntropyOp::forward(logits, targets), "");
}

class CrossEntropyOpBackwardTest : public ::testing::Test {};

TEST_F(CrossEntropyOpBackwardTest, GradientMatchesSoftmaxMinusTarget) {
    auto logits  = make_tensor({2, 3}, {2.f, 0.f, -1.f, 1.f, 3.f, 0.f});
    auto targets = make_tensor({2, 3}, {1.f, 0.f, 0.f, 0.f, 1.f, 0.f});
    auto grad    = make_tensor({1}, {1.f});

    auto grads = CrossEntropyOp::backward(grad, logits, targets);
    ASSERT_EQ(grads.size(), 1u);

    for (size_t i = 0; i < 2; ++i) {
        auto sm = softmax_row(logits, i);
        for (size_t j = 0; j < 3; ++j) {
            const float expected = (sm[j] - targets(i, j)) / 2.f;  // mean over N=2
            EXPECT_NEAR(grads[0](i, j), expected, 1e-6f);
        }
    }
}

TEST_F(CrossEntropyOpBackwardTest, UpstreamGradientScalesAllEntries) {
    auto logits  = make_tensor({1, 3}, {1.f, 2.f, 3.f});
    auto targets = make_tensor({1, 3}, {0.f, 0.f, 1.f});

    auto g1 = CrossEntropyOp::backward(make_tensor({1}, {1.f}), logits, targets);
    auto g2 = CrossEntropyOp::backward(make_tensor({1}, {2.5f}), logits, targets);

    for (size_t j = 0; j < 3; ++j)
        EXPECT_NEAR(g2[0](0, j), 2.5f * g1[0](0, j), 1e-6f);
}

TEST_F(CrossEntropyOpBackwardTest, PerSampleGradientSumsToZeroForOneHotTarget) {
    auto logits  = make_tensor({2, 4}, {2.f, 1.f, 0.f, -1.f, 0.f, 1.f, 2.f, 3.f});
    auto targets = make_tensor({2, 4}, {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f});
    auto grads = CrossEntropyOp::backward(make_tensor({1}, {1.f}), logits, targets);

    for (size_t i = 0; i < 2; ++i) {
        float sum = 0.f;
        for (size_t j = 0; j < 4; ++j)
            sum += grads[0](i, j);
        EXPECT_NEAR(sum, 0.f, 1e-6f);
    }
}

class CrossEntropyFuncTest : public ::testing::Test {};

TEST_F(CrossEntropyFuncTest, NoRequiresGradProducesNoMeta) {
    auto logits  = make_tensor({1, 2}, {1.f, -1.f});
    auto targets = make_tensor({1, 2}, {1.f, 0.f});

    auto loss = cross_entropy(logits, targets);
    EXPECT_EQ(loss.autograd_meta, nullptr);
}

TEST_F(CrossEntropyFuncTest, LogitsRequiresGradProducesMeta) {
    auto logits  = make_tensor({1, 2}, {1.f, -1.f}, /*requires_grad=*/true);
    auto targets = make_tensor({1, 2}, {1.f, 0.f});

    auto loss = cross_entropy(logits, targets);
    EXPECT_TRUE(loss.requires_grad());
}

TEST_F(CrossEntropyFuncTest, TargetsRequiresGradDoesNotCreateGraph) {
    auto logits  = make_tensor({1, 2}, {1.f, -1.f});
    auto targets = make_tensor({1, 2}, {1.f, 0.f}, /*requires_grad=*/true);

    auto loss = cross_entropy(logits, targets);
    EXPECT_EQ(loss.autograd_meta, nullptr);
}

class CrossEntropyAutogradTest : public ::testing::Test {};

TEST_F(CrossEntropyAutogradTest, BackwardPopulatesLogitsGradient) {
    auto logits  = make_tensor({2, 3}, {2.f, 0.f, -1.f, 1.f, 3.f, 0.f}, /*requires_grad=*/true);
    auto targets = make_tensor({2, 3}, {1.f, 0.f, 0.f, 0.f, 1.f, 0.f});

    auto loss = cross_entropy(logits, targets);
    backward(loss);

    ASSERT_TRUE(logits.has_grad());

    for (size_t i = 0; i < 2; ++i) {
        auto sm = softmax_row(logits, i);
        for (size_t j = 0; j < 3; ++j) {
            const float expected = (sm[j] - targets(i, j)) / 2.f;
            EXPECT_NEAR(logits.grad()(i, j), expected, 1e-6f);
        }
    }
}
