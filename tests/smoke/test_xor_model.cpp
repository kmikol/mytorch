// tests/test_xor.cpp
//
// End-to-end integration test: train a small MLP to learn the XOR function.
//
// This test sits at the top of the test pyramid — it exercises the entire
// stack in one shot: Linear, sigmoid, mse_loss, backward, and SGD.  If any
// of those components is broken, this test will fail, but it will not tell
// you which component is at fault.  The unit tests for each component do that.
//
// Design decisions:
//
//   16 samples instead of 4
//     The classic XOR truth table has only 4 unique inputs.  We tile it
//     4× to give 16 samples so the test structure naturally extends to a
//     mini-batch training loop later without changing the dataset code.
//     All 16 samples are still drawn from the same 4 logical patterns —
//     XOR is a function so there is no additional information, but the
//     training loop already iterates over a vector of samples rather than
//     a hard-coded 4-element loop.
//
//   Early stopping
//     Training stops as soon as the mean loss over a full epoch drops below
//     CONVERGENCE_THRESHOLD, rather than always running to MAX_EPOCHS.
//     The convergence epoch is recorded and reported in the test output.
//     The CHECK is that convergence happened within MAX_EPOCHS — not that
//     exactly N epochs were needed, which would be brittle.
//
//   Deterministic RNG seed
//     global_rng is reset to a fixed seed before constructing the network
//     so the test is reproducible across runs and platforms.

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <limits>
#include "tensorlib.h"
#include "layers/linear.h"
#include "optim.h"
#include "ops/ops.h"
#include "activations/activations.h"
#include "loss_functions/mse.h"

// ── constants ────────────────────────────────────────────────────────────────

// Loss below this value on a full epoch is considered converged.
static constexpr float CONVERGENCE_THRESHOLD = 0.01f;

// The test fails if convergence is not reached within this many epochs.
// 2000 is generous — with seed 1337 and lr=1.0 the network typically
// converges well before epoch 1000.
static constexpr int MAX_EPOCHS = 2000;

// Classification threshold: prediction > 0.5 → class 1, else class 0.
static constexpr float CLASSIFICATION_THRESHOLD = 0.5f;

// ── network definition ───────────────────────────────────────────────────────

// XORNet is defined here rather than in a header because it is only used
// in this test file.  Defining it inside the fixture's SetUp would require
// storing it by value in the fixture, which complicates the type.
// A plain struct at file scope is the simplest option.
struct XORNet {
    Linear first_layer{2, 4};    // 2 inputs  → 4 hidden units
    Linear second_layer{4, 1};   // 4 hidden  → 1 output

    Tensor forward(const Tensor& input) const {
        Tensor hidden_activations = sigmoid(first_layer.forward(input));
        Tensor output_activation  = sigmoid(second_layer.forward(hidden_activations));
        return output_activation;
    }

    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> all_parameters;
        for (Tensor* parameter : first_layer.parameters()) {
            all_parameters.push_back(parameter);
        }
        for (Tensor* parameter : second_layer.parameters()) {
            all_parameters.push_back(parameter);
        }
        return all_parameters;
    }
};

// ── dataset helpers ──────────────────────────────────────────────────────────

// XOR truth table tiled to produce num_tiles * 4 samples.
// The four canonical patterns are repeated in order so the dataset
// is balanced and covers all logical inputs regardless of tile count.
//
// Returns a pair of parallel vectors:
//   inputs[i]  — {x0, x1} for sample i
//   targets[i] — XOR(x0, x1) for sample i
static std::pair<std::vector<std::vector<float>>, std::vector<float>>
make_xor_dataset(int num_tiles) {
    // base truth table — all four XOR patterns
    const std::vector<std::vector<float>> base_inputs  = {{0,0}, {0,1}, {1,0}, {1,1}};
    const std::vector<float>              base_targets  = {  0,     1,     1,     0  };

    std::vector<std::vector<float>> inputs;
    std::vector<float>              targets;

    inputs.reserve(base_inputs.size() * static_cast<size_t>(num_tiles));
    targets.reserve(base_targets.size() * static_cast<size_t>(num_tiles));

    for (int tile = 0; tile < num_tiles; ++tile) {
        for (size_t pattern = 0; pattern < base_inputs.size(); ++pattern) {
            inputs.push_back(base_inputs[pattern]);
            targets.push_back(base_targets[pattern]);
        }
    }

    return {inputs, targets};
}

// ── fixture ──────────────────────────────────────────────────────────────────

class XORTest : public ::testing::Test {
protected:
    void SetUp() override {
        // fix the RNG so weight initialisation is deterministic
        global_rng = std::mt19937(1337);

        // 16 samples = 4 canonical XOR patterns tiled 4×
        std::tie(sample_inputs, sample_targets) = make_xor_dataset(4);
    }

    std::vector<std::vector<float>> sample_inputs;
    std::vector<float>              sample_targets;
};

// ════════════════════════════════════════════════════════════════════════════
// Dataset sanity checks
//
// These run before training to confirm the dataset itself is correct.
// A bug in make_xor_dataset() would cause every training test to fail with
// a confusing "did not converge" message — these tests give a clearer signal.
// ════════════════════════════════════════════════════════════════════════════

TEST_F(XORTest, DatasetHasSixteenSamples) {
    EXPECT_EQ(sample_inputs.size(),  16u);
    EXPECT_EQ(sample_targets.size(), 16u);
}

TEST_F(XORTest, DatasetContainsAllFourXORPatterns) {
    // Every canonical pattern must appear at least once.
    bool found_00 = false, found_01 = false, found_10 = false, found_11 = false;

    for (size_t index = 0; index < sample_inputs.size(); ++index) {
        float x0 = sample_inputs[index][0];
        float x1 = sample_inputs[index][1];
        float y  = sample_targets[index];

        if (x0 == 0.f && x1 == 0.f) { EXPECT_FLOAT_EQ(y, 0.f); found_00 = true; }
        if (x0 == 0.f && x1 == 1.f) { EXPECT_FLOAT_EQ(y, 1.f); found_01 = true; }
        if (x0 == 1.f && x1 == 0.f) { EXPECT_FLOAT_EQ(y, 1.f); found_10 = true; }
        if (x0 == 1.f && x1 == 1.f) { EXPECT_FLOAT_EQ(y, 0.f); found_11 = true; }
    }

    EXPECT_TRUE(found_00) << "pattern [0,0] missing from dataset";
    EXPECT_TRUE(found_01) << "pattern [0,1] missing from dataset";
    EXPECT_TRUE(found_10) << "pattern [1,0] missing from dataset";
    EXPECT_TRUE(found_11) << "pattern [1,1] missing from dataset";
}

TEST_F(XORTest, DatasetIsBalanced) {
    // Equal number of class-0 and class-1 samples.
    // An imbalanced dataset would bias the network and make convergence
    // comparisons between runs meaningless.
    int num_class_zero = 0, num_class_one = 0;

    for (float target : sample_targets) {
        if (target == 0.f) ++num_class_zero;
        if (target == 1.f) ++num_class_one;
    }

    EXPECT_EQ(num_class_zero, num_class_one)
        << "dataset must have equal class-0 and class-1 samples";
}

TEST_F(XORTest, EachInputHasExactlyTwoFeatures) {
    for (size_t index = 0; index < sample_inputs.size(); ++index) {
        EXPECT_EQ(sample_inputs[index].size(), 2u)
            << "sample " << index << " does not have 2 features";
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Network architecture checks
//
// Verify the network is wired correctly before we waste time training it.
// A wrong layer size or missing requires_grad would produce a silent failure
// during training rather than a clear error message.
// ════════════════════════════════════════════════════════════════════════════

TEST_F(XORTest, NetworkHasCorrectParameterCount) {
    XORNet network;
    std::vector<Tensor*> parameters = network.parameters();

    // l1: W[4,2] and b[4,1]  →  2 parameter tensors
    // l2: W[1,4] and b[1,1]  →  2 parameter tensors
    // total: 4 parameter tensors
    EXPECT_EQ(parameters.size(), 4u);
}

TEST_F(XORTest, AllParametersRequireGrad) {
    XORNet network;
    std::vector<Tensor*> parameters = network.parameters();

    for (size_t index = 0; index < parameters.size(); ++index) {
        EXPECT_TRUE(parameters[index]->requires_grad())
            << "parameter " << index << " does not require grad";
    }
}

TEST_F(XORTest, ForwardPassProducesCorrectOutputShape) {
    XORNet network;
    Tensor input  = Tensor::from_data({0.f, 1.f}, {2, 1});
    Tensor output = network.forward(input);

    // output must be a scalar wrapped in [1,1] — the shape mse_loss expects
    EXPECT_EQ(output.shape(0), 1);
    EXPECT_EQ(output.shape(1), 1);
}

TEST_F(XORTest, ForwardPassOutputIsInZeroOneRange) {
    // sigmoid output must always lie strictly inside (0, 1)
    XORNet network;

    for (const std::vector<float>& input_values : sample_inputs) {
        Tensor input  = Tensor::from_data(input_values, {2, 1});
        Tensor output = network.forward(input);
        float  value  = output.at(0, 0);

        EXPECT_GT(value, 0.f) << "sigmoid output must be > 0";
        EXPECT_LT(value, 1.f) << "sigmoid output must be < 1";
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Training convergence
//
// The main integration test.  We train with early stopping and assert that
// the network converged within the epoch budget.  The test records and
// prints the convergence epoch for visibility in CI logs.
// ════════════════════════════════════════════════════════════════════════════

TEST_F(XORTest, ConvergesWithinEpochBudget) {
    XORNet network;
    SGD    optimiser(network.parameters(), /*lr=*/1.0f);

    float last_epoch_loss    = std::numeric_limits<float>::max();
    int   convergence_epoch  = -1;

    for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
        float total_loss_this_epoch = 0.f;

        for (size_t sample_index = 0; sample_index < sample_inputs.size(); ++sample_index) {
            Tensor input  = Tensor::from_data(sample_inputs[sample_index],     {2, 1});
            Tensor target = Tensor::from_data({sample_targets[sample_index]},  {1, 1});

            Tensor prediction = network.forward(input);
            Tensor loss       = mse_loss(prediction, target);

            total_loss_this_epoch += loss.at(0, 0);

            optimiser.zero_grad();
            backward(loss);
            optimiser.step();
        }

        last_epoch_loss = total_loss_this_epoch / static_cast<float>(sample_inputs.size());

        // early stopping — record epoch and exit training loop
        if (last_epoch_loss < CONVERGENCE_THRESHOLD) {
            convergence_epoch = epoch;
            break;
        }
    }

    // report convergence details regardless of pass/fail so CI logs are useful
    if (convergence_epoch >= 0) {
        printf("\n  Converged at epoch %d  (loss = %.6f)\n",
               convergence_epoch, last_epoch_loss);
    } else {
        printf("\n  Did not converge within %d epochs  (final loss = %.6f)\n",
               MAX_EPOCHS, last_epoch_loss);
    }

    EXPECT_GE(convergence_epoch, 0)
        << "network did not converge within " << MAX_EPOCHS << " epochs; "
        << "final loss = " << last_epoch_loss;
}


TEST_F(XORTest, ConvergesWithinEpochBudgetBatched) {
    XORNet network;
    SGD    optimiser(network.parameters(), /*lr=*/5.0f);

    float last_epoch_loss    = std::numeric_limits<float>::max();
    int   convergence_epoch  = -1;

    for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {

        // correct: walks feature by feature → each row is one feature
        int num_features = sample_inputs[0].size();   // 2
        int num_samples  = sample_inputs.size();       // 4

        std::vector<float> flat_inputs;

        // outer loop: one row per feature
        for (int f = 0; f < num_features; f++) {
            // inner loop: that feature's value for every sample
            for (int s = 0; s < num_samples; s++) {
                flat_inputs.push_back(sample_inputs[s][f]);
            }
        }

        Tensor batch_inputs  = Tensor::from_data(flat_inputs, {2, static_cast<int64_t>(sample_inputs.size())});
        Tensor batch_targets = Tensor::from_data(sample_targets,{1, static_cast<int64_t>(sample_targets.size())});

        Tensor prediction = network.forward(batch_inputs);
        Tensor loss       = mse_loss(prediction, batch_targets);

        last_epoch_loss = loss.at(0, 0);

        optimiser.zero_grad();
        backward(loss);
        optimiser.step();

        // early stopping — record epoch and exit training loop
        if (last_epoch_loss < CONVERGENCE_THRESHOLD) {
            convergence_epoch = epoch;
            break;
        }
    }

    // report convergence details regardless of pass/fail so CI logs are useful
    if (convergence_epoch >= 0) {
        printf("\n  Converged at epoch %d  (loss = %.6f)\n",
               convergence_epoch, last_epoch_loss);
    } else {
        printf("\n  Did not converge within %d epochs  (final loss = %.6f)\n",
               MAX_EPOCHS, last_epoch_loss);
    }

    EXPECT_GE(convergence_epoch, 0)
        << "network did not converge within " << MAX_EPOCHS << " epochs; "
        << "final loss = " << last_epoch_loss;
}

TEST_F(XORTest, FinalLossIsBelowThreshold) {
    // Separate from convergence epoch check — verifies the loss value itself
    // rather than just that early stopping triggered.
    XORNet network;
    SGD    optimiser(network.parameters(), /*lr=*/1.0f);
    float  last_epoch_loss = std::numeric_limits<float>::max();

    for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
        float total_loss_this_epoch = 0.f;

        for (size_t sample_index = 0; sample_index < sample_inputs.size(); ++sample_index) {
            Tensor input  = Tensor::from_data(sample_inputs[sample_index],    {2, 1});
            Tensor target = Tensor::from_data({sample_targets[sample_index]}, {1, 1});

            Tensor prediction = network.forward(input);
            Tensor loss       = mse_loss(prediction, target);
            total_loss_this_epoch += loss.at(0, 0);

            optimiser.zero_grad();
            backward(loss);
            optimiser.step();
        }

        last_epoch_loss = total_loss_this_epoch / static_cast<float>(sample_inputs.size());
        if (last_epoch_loss < CONVERGENCE_THRESHOLD) break;
    }

    EXPECT_LT(last_epoch_loss, CONVERGENCE_THRESHOLD)
        << "final loss " << last_epoch_loss
        << " did not fall below threshold " << CONVERGENCE_THRESHOLD;
}