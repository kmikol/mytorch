// tests/test_cross_entropy.cpp
//
// End-to-end integration test: train a small MLP to learn a toy 3-class
// classification task using cross_entropy_loss.
//
// Mirrors the structure and conventions of test_xor.cpp so the two files
// can be read side-by-side.  The key differences are:
//
//   Loss function
//     mse_loss → cross_entropy_loss.  The network now outputs raw logits
//     (no output sigmoid) because cross_entropy_loss fuses softmax internally.
//
//   Output shape
//     Scalar [1,1] → [C, N] = [3, 1] per sample, or [3, batch_size] batched.
//
//   Dataset — "corner classification"
//     Four input patterns at the corners / centre of the unit square,
//     each assigned a distinct class so the problem is non-linearly separable
//     and strictly requires a hidden layer to solve:
//
//       [0, 0]  →  class 0
//       [1, 0]  →  class 1
//       [0, 1]  →  class 2
//       [1, 1]  →  class 0   ← same as [0,0], breaks linear separability
//
//     This is structurally analogous to XOR (same inputs, same asymmetry).
//
//   Evaluation metric
//     In addition to the loss threshold, the final trained model is evaluated
//     on classification accuracy — the argmax of the output logits is compared
//     to the ground-truth label.  We require 100 % accuracy on training data
//     after convergence.
//
//   Deterministic RNG seed
//     Same convention as test_xor.cpp: global_rng is reset to a fixed seed
//     before constructing each network so tests are reproducible.

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include "tensorlib.h"
#include "layers/linear.h"
#include "optim.h"
#include "ops/ops.h"
#include "loss_functions/cross_entropy.h"

// ── constants ────────────────────────────────────────────────────────────────

// Number of output classes in the toy dataset.
static constexpr int   NUM_CLASSES           = 3;

// Loss below this value on a full epoch is considered converged.
// Cross-entropy loss is in nats, not [0,1] squared error, so the threshold
// is set lower than the MSE threshold used in test_xor.cpp.
static constexpr float CONVERGENCE_THRESHOLD = 0.05f;

// The test fails if convergence is not reached within this many epochs.
static constexpr int   MAX_EPOCHS            = 3000;

// ── network definition ───────────────────────────────────────────────────────

// CornerNet maps 2D inputs to 3 raw logits.
// No sigmoid on the output — cross_entropy_loss applies softmax internally.
struct CornerNet {
    Linear first_layer{2, 8};              // 2 inputs  → 8 hidden units
    Linear second_layer{8, NUM_CLASSES};   // 8 hidden  → 3 logits

    // Returns logits with shape [NUM_CLASSES, batch_size].
    Tensor forward(const Tensor& input) const {
        Tensor hidden = sigmoid(first_layer.forward(input));
        Tensor logits = second_layer.forward(hidden);   // no activation — raw scores
        return logits;
    }

    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> all_parameters;
        for (Tensor* p : first_layer.parameters())  all_parameters.push_back(p);
        for (Tensor* p : second_layer.parameters()) all_parameters.push_back(p);
        return all_parameters;
    }
};

// ── dataset helpers ──────────────────────────────────────────────────────────

// Toy dataset: four corner / centre patterns, non-linearly separable.
//
//   [0,0] → class 0     [1,0] → class 1
//   [0,1] → class 2     [1,1] → class 0   ← same label as [0,0]
//
// Tiled num_tiles times so the training loop iterates a vector of samples,
// matching the structure used in the XOR test.
//
// Returns:
//   inputs[i]  — {x0, x1} feature pair for sample i
//   targets[i] — integer class index (stored as float) for sample i
static std::pair<std::vector<std::vector<float>>, std::vector<float>>
make_corner_dataset(int num_tiles) {
    const std::vector<std::vector<float>> base_inputs   = {{0,0}, {1,0}, {0,1}, {1,1}};
    const std::vector<float>              base_targets   = {  0,     1,     2,     0  };

    std::vector<std::vector<float>> inputs;
    std::vector<float>              targets;

    inputs.reserve(base_inputs.size()   * static_cast<size_t>(num_tiles));
    targets.reserve(base_targets.size() * static_cast<size_t>(num_tiles));

    for (int tile = 0; tile < num_tiles; ++tile) {
        for (size_t pattern = 0; pattern < base_inputs.size(); ++pattern) {
            inputs.push_back(base_inputs[pattern]);
            targets.push_back(base_targets[pattern]);
        }
    }
    return {inputs, targets};
}

// Returns the argmax index over the first dimension (class axis) for column n.
// logits must have shape [C, N]; returns the predicted class for sample n.
static int argmax_class(const Tensor& logits, int64_t n) {
    int64_t C     = logits.shape(0);
    int     best  = 0;
    float   best_val = logits.at(0, n);
    for (int64_t c = 1; c < C; ++c) {
        float v = logits.at(c, n);
        if (v > best_val) { best_val = v; best = static_cast<int>(c); }
    }
    return best;
}

// ── fixture ──────────────────────────────────────────────────────────────────

class CrossEntropyTest : public ::testing::Test {
protected:
    void SetUp() override {
        global_rng = std::mt19937(1337);
        std::tie(sample_inputs, sample_targets) = make_corner_dataset(4);
    }

    std::vector<std::vector<float>> sample_inputs;
    std::vector<float>              sample_targets;
};

// ════════════════════════════════════════════════════════════════════════════
// Dataset sanity checks
// ════════════════════════════════════════════════════════════════════════════

TEST_F(CrossEntropyTest, DatasetHasSixteenSamples) {
    EXPECT_EQ(sample_inputs.size(),  16u);
    EXPECT_EQ(sample_targets.size(), 16u);
}

TEST_F(CrossEntropyTest, DatasetContainsAllFourPatterns) {
    bool found_00 = false, found_10 = false, found_01 = false, found_11 = false;

    for (size_t i = 0; i < sample_inputs.size(); ++i) {
        float x0 = sample_inputs[i][0];
        float x1 = sample_inputs[i][1];
        float y  = sample_targets[i];

        if (x0 == 0.f && x1 == 0.f) { EXPECT_FLOAT_EQ(y, 0.f); found_00 = true; }
        if (x0 == 1.f && x1 == 0.f) { EXPECT_FLOAT_EQ(y, 1.f); found_10 = true; }
        if (x0 == 0.f && x1 == 1.f) { EXPECT_FLOAT_EQ(y, 2.f); found_01 = true; }
        if (x0 == 1.f && x1 == 1.f) { EXPECT_FLOAT_EQ(y, 0.f); found_11 = true; }
    }

    EXPECT_TRUE(found_00) << "pattern [0,0] missing";
    EXPECT_TRUE(found_10) << "pattern [1,0] missing";
    EXPECT_TRUE(found_01) << "pattern [0,1] missing";
    EXPECT_TRUE(found_11) << "pattern [1,1] missing";
}

TEST_F(CrossEntropyTest, TargetsAreValidClassIndices) {
    // Every target must be a whole number in [0, NUM_CLASSES).
    for (size_t i = 0; i < sample_targets.size(); ++i) {
        float t = sample_targets[i];
        EXPECT_EQ(t, std::floor(t))
            << "target[" << i << "] is not an integer: " << t;
        EXPECT_GE(t, 0.f)
            << "target[" << i << "] is negative: " << t;
        EXPECT_LT(t, static_cast<float>(NUM_CLASSES))
            << "target[" << i << "] >= NUM_CLASSES: " << t;
    }
}

TEST_F(CrossEntropyTest, DatasetHasAllThreeClasses) {
    // Each class must appear at least once — otherwise the loss is trivially
    // minimised without learning the harder class.
    std::vector<int> class_counts(NUM_CLASSES, 0);
    for (float t : sample_targets) ++class_counts[static_cast<int>(t)];

    for (int c = 0; c < NUM_CLASSES; ++c) {
        EXPECT_GT(class_counts[c], 0)
            << "class " << c << " has no samples in the dataset";
    }
}

TEST_F(CrossEntropyTest, EachInputHasExactlyTwoFeatures) {
    for (size_t i = 0; i < sample_inputs.size(); ++i) {
        EXPECT_EQ(sample_inputs[i].size(), 2u)
            << "sample " << i << " does not have 2 features";
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Network architecture checks
// ════════════════════════════════════════════════════════════════════════════

TEST_F(CrossEntropyTest, NetworkHasCorrectParameterCount) {
    CornerNet network;
    // l1: W[8,2] and b[8,1]  →  2 tensors
    // l2: W[3,8] and b[3,1]  →  2 tensors
    // total: 4 parameter tensors
    EXPECT_EQ(network.parameters().size(), 4u);
}

TEST_F(CrossEntropyTest, AllParametersRequireGrad) {
    CornerNet network;
    std::vector<Tensor*> params = network.parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        EXPECT_TRUE(params[i]->requires_grad())
            << "parameter " << i << " does not require grad";
    }
}

TEST_F(CrossEntropyTest, ForwardPassProducesCorrectOutputShape) {
    // Single-sample forward: input [2,1] → logits [NUM_CLASSES, 1]
    CornerNet network;
    Tensor input  = Tensor::from_data({0.f, 1.f}, {2, 1});
    Tensor logits = network.forward(input);

    EXPECT_EQ(logits.shape(0), NUM_CLASSES);
    EXPECT_EQ(logits.shape(1), 1);
}

TEST_F(CrossEntropyTest, ForwardPassLogitsAreFinite) {
    // Logits (unlike sigmoid outputs) are unbounded, but must not be NaN/Inf
    // at initialisation — that would indicate a weight init problem.
    CornerNet network;

    for (const std::vector<float>& input_vals : sample_inputs) {
        Tensor input  = Tensor::from_data(input_vals, {2, 1});
        Tensor logits = network.forward(input);

        for (int c = 0; c < NUM_CLASSES; ++c) {
            float v = logits.at(c, 0);
            EXPECT_TRUE(std::isfinite(v))
                << "logit[" << c << "] is not finite for input ["
                << input_vals[0] << ", " << input_vals[1] << "]";
        }
    }
}

TEST_F(CrossEntropyTest, CrossEntropyLossIsPositive) {
    // Cross-entropy is always strictly positive (it is -log(p) with p in (0,1)).
    CornerNet network;
    Tensor input  = Tensor::from_data({1.f, 0.f}, {2, 1});
    Tensor target = Tensor::from_data({1.f},       {1, 1});   // class 1
    Tensor logits = network.forward(input);
    Tensor loss   = cross_entropy_loss(logits, target);

    EXPECT_GT(loss.at(0, 0), 0.f)
        << "cross-entropy loss must be strictly positive";
}

TEST_F(CrossEntropyTest, CrossEntropyLossIsFinite) {
    CornerNet network;
    Tensor input  = Tensor::from_data({0.f, 0.f}, {2, 1});
    Tensor target = Tensor::from_data({0.f},       {1, 1});
    Tensor logits = network.forward(input);
    Tensor loss   = cross_entropy_loss(logits, target);

    EXPECT_TRUE(std::isfinite(loss.at(0, 0)))
        << "cross-entropy loss must be finite";
}

// ════════════════════════════════════════════════════════════════════════════
// Training convergence — sample-at-a-time SGD
// ════════════════════════════════════════════════════════════════════════════

TEST_F(CrossEntropyTest, ConvergesWithinEpochBudget) {
    global_rng = std::mt19937(1337);
    CornerNet network;
    SGD       optimiser(network.parameters(), /*lr=*/1.0f);

    float last_epoch_loss   = std::numeric_limits<float>::max();
    int   convergence_epoch = -1;

    for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
        float total_loss = 0.f;

        for (size_t i = 0; i < sample_inputs.size(); ++i) {
            Tensor input  = Tensor::from_data(sample_inputs[i],      {2, 1});
            Tensor target = Tensor::from_data({sample_targets[i]},   {1, 1});

            Tensor logits = network.forward(input);
            Tensor loss   = cross_entropy_loss(logits, target);

            total_loss += loss.at(0, 0);

            optimiser.zero_grad();
            backward(loss);
            optimiser.step();
        }

        last_epoch_loss = total_loss / static_cast<float>(sample_inputs.size());

        if (last_epoch_loss < CONVERGENCE_THRESHOLD) {
            convergence_epoch = epoch;
            break;
        }
    }

    if (convergence_epoch >= 0) {
        printf("\n  [SGD] Converged at epoch %d  (loss = %.6f)\n",
               convergence_epoch, last_epoch_loss);
    } else {
        printf("\n  [SGD] Did not converge within %d epochs  (final loss = %.6f)\n",
               MAX_EPOCHS, last_epoch_loss);
    }

    EXPECT_GE(convergence_epoch, 0)
        << "network did not converge within " << MAX_EPOCHS << " epochs; "
        << "final loss = " << last_epoch_loss;
}

TEST_F(CrossEntropyTest, FinalLossIsBelowThreshold) {
    // Checks the loss value directly, independent of the convergence epoch.
    global_rng = std::mt19937(1337);
    CornerNet network;
    SGD       optimiser(network.parameters(), /*lr=*/1.0f);
    float     last_epoch_loss = std::numeric_limits<float>::max();

    for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
        float total_loss = 0.f;

        for (size_t i = 0; i < sample_inputs.size(); ++i) {
            Tensor input  = Tensor::from_data(sample_inputs[i],     {2, 1});
            Tensor target = Tensor::from_data({sample_targets[i]},  {1, 1});

            Tensor logits = network.forward(input);
            Tensor loss   = cross_entropy_loss(logits, target);
            total_loss   += loss.at(0, 0);

            optimiser.zero_grad();
            backward(loss);
            optimiser.step();
        }

        last_epoch_loss = total_loss / static_cast<float>(sample_inputs.size());
        if (last_epoch_loss < CONVERGENCE_THRESHOLD) break;
    }

    EXPECT_LT(last_epoch_loss, CONVERGENCE_THRESHOLD)
        << "final loss " << last_epoch_loss
        << " did not fall below threshold " << CONVERGENCE_THRESHOLD;
}

// ════════════════════════════════════════════════════════════════════════════
// Training convergence — batched forward pass
//
// All 16 samples packed into a single [2, 16] input tensor.
// Logits returned are [NUM_CLASSES, 16]; loss is the mean over the batch.
// ════════════════════════════════════════════════════════════════════════════

TEST_F(CrossEntropyTest, ConvergesWithinEpochBudgetBatched) {
    global_rng = std::mt19937(1337);
    CornerNet network;
    SGD       optimiser(network.parameters(), /*lr=*/5.0f);

    float last_epoch_loss   = std::numeric_limits<float>::max();
    int   convergence_epoch = -1;

    // Pre-build the batch tensors once — they don't change between epochs.
    int num_features = static_cast<int>(sample_inputs[0].size());   // 2
    int num_samples  = static_cast<int>(sample_inputs.size());       // 16

    // Layout: row-major, one row per feature.
    // flat_inputs[f * num_samples + s] = sample_inputs[s][f]
    std::vector<float> flat_inputs;
    flat_inputs.reserve(num_features * num_samples);
    for (int f = 0; f < num_features; ++f)
        for (int s = 0; s < num_samples; ++s)
            flat_inputs.push_back(sample_inputs[s][f]);

    Tensor batch_inputs  = Tensor::from_data(flat_inputs,    {num_features, num_samples});
    Tensor batch_targets = Tensor::from_data(sample_targets, {1, num_samples});

    for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
        Tensor logits = network.forward(batch_inputs);
        Tensor loss   = cross_entropy_loss(logits, batch_targets);

        last_epoch_loss = loss.at(0, 0);

        optimiser.zero_grad();
        backward(loss);
        optimiser.step();

        if (last_epoch_loss < CONVERGENCE_THRESHOLD) {
            convergence_epoch = epoch;
            break;
        }
    }

    if (convergence_epoch >= 0) {
        printf("\n  [Batched] Converged at epoch %d  (loss = %.6f)\n",
               convergence_epoch, last_epoch_loss);
    } else {
        printf("\n  [Batched] Did not converge within %d epochs  (final loss = %.6f)\n",
               MAX_EPOCHS, last_epoch_loss);
    }

    EXPECT_GE(convergence_epoch, 0)
        << "batched network did not converge within " << MAX_EPOCHS << " epochs; "
        << "final loss = " << last_epoch_loss;
}

// ════════════════════════════════════════════════════════════════════════════
// Classification accuracy after training
//
// A model can in theory produce low average loss while still mis-classifying
// individual samples (e.g. by hedging toward a uniform softmax).  This test
// checks that the argmax prediction is correct for every training sample,
// which is a stricter signal than the loss threshold alone.
// ════════════════════════════════════════════════════════════════════════════

TEST_F(CrossEntropyTest, AchievesPerfectTrainingAccuracy) {
    global_rng = std::mt19937(1337);
    CornerNet network;
    SGD       optimiser(network.parameters(), /*lr=*/1.0f);

    // Train to the same threshold used by the convergence test.
    for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
        float total_loss = 0.f;

        for (size_t i = 0; i < sample_inputs.size(); ++i) {
            Tensor input  = Tensor::from_data(sample_inputs[i],     {2, 1});
            Tensor target = Tensor::from_data({sample_targets[i]},  {1, 1});

            Tensor logits = network.forward(input);
            Tensor loss   = cross_entropy_loss(logits, target);
            total_loss   += loss.at(0, 0);

            optimiser.zero_grad();
            backward(loss);
            optimiser.step();
        }

        float mean_loss = total_loss / static_cast<float>(sample_inputs.size());
        if (mean_loss < CONVERGENCE_THRESHOLD) break;
    }

    // Evaluate argmax predictions on every training sample.
    int correct = 0;
    int total   = static_cast<int>(sample_inputs.size());

    for (size_t i = 0; i < sample_inputs.size(); ++i) {
        Tensor input      = Tensor::from_data(sample_inputs[i], {2, 1});
        Tensor logits     = network.forward(input);
        int    predicted  = argmax_class(logits, 0);
        int    ground_truth = static_cast<int>(sample_targets[i]);

        if (predicted == ground_truth) ++correct;
    }

    float accuracy = static_cast<float>(correct) / static_cast<float>(total);
    printf("\n  Training accuracy: %d / %d  (%.1f %%)\n",
           correct, total, accuracy * 100.f);

    EXPECT_EQ(correct, total)
        << "expected 100% training accuracy after convergence; "
        << "got " << correct << "/" << total;
}

// ════════════════════════════════════════════════════════════════════════════
// Gradient sanity — loss decreases after a single update
//
// A focused unit-level check that sits below the full training tests:
// given a random initialisation and one sample, the loss after one SGD step
// must be strictly lower than the loss before.  This fails immediately if
// backward() or SGD.step() is broken, giving a clearer signal than waiting
// for a convergence test to time out.
// ════════════════════════════════════════════════════════════════════════════

TEST_F(CrossEntropyTest, LossDecreasesAfterOneSGDStep) {
    global_rng = std::mt19937(42);
    CornerNet network;
    SGD       optimiser(network.parameters(), /*lr=*/0.5f);

    // Use the first sample for this check.
    Tensor input  = Tensor::from_data(sample_inputs[0],     {2, 1});
    Tensor target = Tensor::from_data({sample_targets[0]},  {1, 1});

    // Loss before update.
    float loss_before = cross_entropy_loss(network.forward(input), target).at(0, 0);

    // One SGD step.
    Tensor logits = network.forward(input);
    Tensor loss   = cross_entropy_loss(logits, target);
    optimiser.zero_grad();
    backward(loss);
    optimiser.step();

    // Loss after update.
    float loss_after = cross_entropy_loss(network.forward(input), target).at(0, 0);

    EXPECT_LT(loss_after, loss_before)
        << "loss did not decrease after one SGD step: "
        << "before=" << loss_before << "  after=" << loss_after;
}