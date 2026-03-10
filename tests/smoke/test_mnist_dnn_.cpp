// tests/mnist_smoke_test.cpp
#include <gtest/gtest.h>
#include <cstdio>
#include "dataset/mnist_dataset.h"
#include "dataset/dataloader.h"
#include "networks/mnist_dnn.h"
#include "loss_functions/cross_entropy.h"
#include "ops/ops.h"
#include "optim.h"

// path to MNIST binary files — override via environment variable
// e.g. MNIST_PATH=/data/mnist ./tests
static std::string mnist_path() {
    return "data/MNIST_subsamp1000";
}

// ----------------------------------------------------------------
// helper: compute accuracy over one full pass of a dataloader
// returns fraction of correctly classified samples in [0, 1]
// ----------------------------------------------------------------
static float evaluate_accuracy(MnistDNN& net, DataLoader& loader) {
    loader.reset();

    int64_t correct = 0;
    int64_t total   = 0;

    while (loader.has_next()) {
        auto [inputs, targets] = loader.next_batch();
        // inputs:  [784, N]
        // targets: [10,  N] — one-hot

        Tensor logits = net.forward(inputs);   // [10, N]
        int64_t N     = logits.shape(1);

        for (int64_t n = 0; n < N; n++) {

            // predicted class: argmax over logits for sample n
            int64_t pred_class = 0;
            float   best_val   = logits.at(0, n);
            for (int64_t c = 1; c < 10; c++) {
                float v = logits.at(c, n);
                if (v > best_val) { best_val = v; pred_class = c; }
            }

            // true class: argmax over one-hot target for sample n
            int64_t true_class = 0;
            for (int64_t c = 1; c < 10; c++)
                if (targets.at(c, n) > targets.at(true_class, n))
                    true_class = c;

            if (pred_class == true_class) ++correct;
            ++total;
        }
    }

    return (float)correct / (float)total;
}

// ----------------------------------------------------------------
// helper: build class-index targets [1, N] from one-hot [10, N]
// cross_entropy_loss expects target class indices, not one-hot
// ----------------------------------------------------------------
static Tensor onehot_to_indices(const Tensor& onehot) {
    // onehot: [10, N]
    int64_t N = onehot.shape(1);
    std::vector<float> indices(N);
    for (int64_t n = 0; n < N; n++) {
        for (int64_t c = 0; c < 10; c++) {
            if (onehot.at(c, n) > 0.5f) {
                indices[n] = (float)c;
                break;
            }
        }
    }
    return Tensor::from_data(indices, {1, N});
}



// ----------------------------------------------------------------
// minimal sanity: single forward pass produces the right output shape
// no training, no files needed
// ----------------------------------------------------------------
TEST(MnistSmoke, ForwardOutputShape) {
    global_rng = std::mt19937(42);
    MnistDNN net(784, 10);

    // fake batch of 8 images
    Tensor x = Tensor::zeros({784, 8});
    Tensor out = net.forward(x);

    // output should be [10, 8] — 10 class logits per sample
    EXPECT_EQ(out.shape(0), 10);
    EXPECT_EQ(out.shape(1), 8);
}

TEST(MnistSmoke, ForwardOutputIsFinite) {
    global_rng = std::mt19937(42);
    MnistDNN net(784, 10);

    Tensor x   = Tensor::zeros({784, 4});
    Tensor out = net.forward(x);

    for (int64_t c = 0; c < 10; c++)
        for (int64_t n = 0; n < 4; n++) {
            EXPECT_FALSE(std::isnan(out.at(c, n))) << "nan at [" << c << "," << n << "]";
            EXPECT_FALSE(std::isinf(out.at(c, n))) << "inf at [" << c << "," << n << "]";
        }
}

TEST(MnistSmoke, ParametersReceiveGradients) {
    // single forward+backward — all parameters should have gradients
    global_rng = std::mt19937(42);
    MnistDNN net(784, 10);
    SGD optim(net.parameters(), 0.01f);

    Tensor x       = Tensor::zeros({784, 4});
    Tensor targets = Tensor::from_data({0.f, 1.f, 2.f, 3.f}, {1, 4});

    Tensor logits = net.forward(x);
    Tensor loss   = cross_entropy_loss(logits, targets);

    optim.zero_grad();
    backward(loss);

    for (Tensor* p : net.parameters()) {
        EXPECT_TRUE(p->has_grad())
            << "parameter should have a gradient after backward";
    }
}

TEST(MnistDebug, InputsAreNormalised) {
    std::string path = mnist_path();
    std::string images = path + "/train-images-idx3-ubyte";
    std::string labels = path + "/train-labels-idx1-ubyte";
    {
        std::ifstream f(images);
        if (!f.good()) GTEST_SKIP() << "MNIST files not found";
    }

    MNISTDataset dataset(images, labels);
    DataLoader   loader(dataset, 256, false);
    auto [inputs, targets] = loader.next_batch();

    // find min, max, mean of the input batch
    float min_val  =  1e9f;
    float max_val  = -1e9f;
    float sum      =  0.f;
    int64_t total  = inputs.shape(0) * inputs.shape(1);

    for (int64_t row = 0; row < inputs.shape(0); ++row) {
        for (int64_t col = 0; col < inputs.shape(1); ++col) {
            float value = inputs.at(row, col);
            min_val  = std::min(min_val, value);
            max_val  = std::max(max_val, value);
            sum     += value;
        }
    }
    float mean = sum / static_cast<float>(total);

    printf("\n  input stats: min=%.3f  max=%.3f  mean=%.3f\n",
           min_val, max_val, mean);

    // values should be in roughly [0,1] or [-1,1] after normalisation
    // if max is ~255 the dataset loader is not normalising
    EXPECT_LE(max_val, 1.0f)
        << "inputs appear unnormalised (max=" << max_val
        << ") — divide by 255 in the dataset loader";
    EXPECT_GE(min_val, -1.0f);
}

TEST(MnistDebug, OverfitsSingleBatch) {
    std::string path = mnist_path();
    std::string images = path + "/train-images-idx3-ubyte";
    std::string labels = path + "/train-labels-idx1-ubyte";
    {
        std::ifstream f(images);
        if (!f.good()) GTEST_SKIP() << "MNIST files not found at " << path;
    }

    // pull exactly one batch from real data and never advance the loader again
    MNISTDataset dataset(images, labels);
    DataLoader   loader(dataset, 8, /*shuffle=*/false);
    auto [inputs, targets_onehot] = loader.next_batch();
    Tensor targets = onehot_to_indices(targets_onehot);

    // print the true labels so we know what the batch contains
    printf("\n  True labels: ");
    for (int64_t n = 0; n < targets.shape(1); ++n)
        printf("%.0f ", targets.at(0, n));
    printf("\n\n");

    // print size, num cols and num rows of the input
    printf("  input shape: [%lld, %lld]\n",
           inputs.shape(0), inputs.shape(1));
    printf("  dataset image rows: %u\n", dataset.image_rows());
    printf("  dataset image cols: %u\n", dataset.image_cols());

    global_rng = std::mt19937(42);
    MnistDNN net(dataset.input_size(), dataset.num_classes());
    SGD optim(net.parameters(), 0.1f);

    const int NUM_STEPS = 200;

    float initial_loss = 0.f;
    float final_loss   = 0.f;

    for (int step = 0; step < NUM_STEPS; ++step) {
        Tensor logits = net.forward(inputs);
        Tensor loss   = cross_entropy_loss(logits, targets);
        float  loss_value = loss.at(0, 0);

        if (step == 0)            initial_loss = loss_value;
        if (step == NUM_STEPS-1)  final_loss   = loss_value;

        optim.zero_grad();
        backward(loss);
        optim.step();
    }


    EXPECT_LT(final_loss, 0.1f)
        << "cannot overfit 8 real samples after " << NUM_STEPS << " steps"
        << " — initial loss was " << initial_loss;
}





// tests/mnist_training_debug.cpp
//
// Targeted tests for the full training loop failure mode:
// single-batch overfit works, full training stalls.
//
// The four tests below are ordered from most to least likely cause.
// Run them one at a time and stop at the first failure — that is the bug.
//
//   make smoke_tests filter="MnistTrainingDebug*"

// ════════════════════════════════════════════════════════════════════════════
// 1.  zero_grad() actually zeros all gradients
//
// This is the most common cause of the "overfits one batch, stalls on full
// dataset" failure.  If zero_grad() is a no-op or only zeros some params,
// gradients from previous batches accumulate.  On batch k, the effective
// gradient is the sum of gradients from batches 1..k — the signal from the
// current batch gets buried under history, converging to the mean gradient
// which is nearly zero for a balanced dataset.
// ════════════════════════════════════════════════════════════════════════════

TEST(MnistTrainingDebug, ZeroGradActuallyClearsAllGradients) {
    global_rng = std::mt19937(42);
    MnistDNN net(784, 10);
    SGD      optimiser(net.parameters(), 0.01f);

    // run two forward+backward passes without zeroing between them
    // then zero and check — everything must be exactly zero
    Tensor inputs_a  = Tensor::zeros({784, 4});
    Tensor targets_a = Tensor::from_data({0.f, 1.f, 2.f, 3.f}, {1, 4});

    Tensor inputs_b  = Tensor::zeros({784, 4});
    Tensor targets_b = Tensor::from_data({4.f, 5.f, 6.f, 7.f}, {1, 4});

    // first pass
    Tensor logits_a = net.forward(inputs_a);
    Tensor loss_a   = cross_entropy_loss(logits_a, targets_a);
    backward(loss_a);

    // second pass WITHOUT zeroing — gradients accumulate
    Tensor logits_b = net.forward(inputs_b);
    Tensor loss_b   = cross_entropy_loss(logits_b, targets_b);
    backward(loss_b);

    // verify gradients are non-zero before zeroing (proves backward ran)
    bool any_nonzero_before = false;
    for (Tensor* param : net.parameters()) {
        if (!param->has_grad()) continue;
        for (int64_t row = 0; row < param->grad().shape(0); ++row) {
            for (int64_t col = 0; col < param->grad().shape(1); ++col) {
                if (std::abs(param->grad().at(row, col)) > 1e-9f) {
                    any_nonzero_before = true;
                }
            }
        }
    }
    ASSERT_TRUE(any_nonzero_before)
        << "gradients were already zero before zero_grad() — backward() may be broken";

    // NOW zero and check every element is exactly zero
    optimiser.zero_grad();

    printf("\n  Checking all %zu parameters are zeroed:\n", net.parameters().size());

    for (size_t param_index = 0; param_index < net.parameters().size(); ++param_index) {
        Tensor* param = net.parameters()[param_index];

        if (!param->has_grad()) {
            printf("    param %zu: no grad tensor (skipped)\n", param_index);
            continue;
        }

        float max_abs_value = 0.f;
        int64_t nonzero_count = 0;

        for (int64_t row = 0; row < param->grad().shape(0); ++row) {
            for (int64_t col = 0; col < param->grad().shape(1); ++col) {
                float value = param->grad().at(row, col);
                max_abs_value = std::max(max_abs_value, std::abs(value));
                if (std::abs(value) > 1e-9f) ++nonzero_count;
            }
        }

        printf("    param %zu  shape [%lld,%lld]  max_abs=%.2e  nonzero=%lld\n",
               param_index,
               param->grad().shape(0),
               param->grad().shape(1),
               max_abs_value,
               nonzero_count);

        EXPECT_EQ(nonzero_count, 0)
            << "param " << param_index
            << " has " << nonzero_count << " nonzero gradient elements after zero_grad()";
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 2.  Loss is averaged over batch size, not summed
//
// If cross_entropy_loss sums over the batch rather than averaging, a batch
// of 256 samples produces a gradient 256× larger than a batch of 1.
// The single-batch overfit used 8 samples and lr=0.1 — effective step 0.8.
// The full training loop uses 256 samples and lr=0.5 — effective step 128.
// That is exploding gradients hiding behind a "reasonable" lr.
//
// For a correct implementation: loss(batch_of_N) ≈ loss(single_sample)
// because both are per-sample averages.
// ════════════════════════════════════════════════════════════════════════════

TEST(MnistTrainingDebug, LossIsAveragedOverBatchSizeNotSummed) {
    global_rng = std::mt19937(42);
    MnistDNN net_single(784, 10);

    global_rng = std::mt19937(42);
    MnistDNN net_batch(784, 10);

    // identical input repeated — single sample vs batch of 32
    std::vector<float> single_pixel(784, 0.5f);
    std::vector<float> batch_pixels(784 * 32, 0.5f);   // same pixel, 32 times

    Tensor single_input  = Tensor::from_data(single_pixel,  {784, 1});
    Tensor single_target = Tensor::from_data({3.f},          {1,   1});

    Tensor batch_input   = Tensor::from_data(batch_pixels,   {784, 32});

    std::vector<float> batch_target_values(32, 3.f);
    Tensor batch_target  = Tensor::from_data(batch_target_values, {1, 32});

    Tensor loss_single = cross_entropy_loss(net_single.forward(single_input),  single_target);
    Tensor loss_batch  = cross_entropy_loss(net_batch.forward(batch_input),    batch_target);

    float single_value = loss_single.at(0, 0);
    float batch_value  = loss_batch.at(0, 0);

    printf("\n  loss on 1 sample:  %.6f\n", single_value);
    printf("  loss on 32 copies: %.6f\n",  batch_value);
    printf("  ratio (batch/single): %.4f  (should be ~1.0 if averaged, ~32.0 if summed)\n",
           batch_value / single_value);

    // if the implementation averages: ratio ≈ 1.0
    // if the implementation sums:     ratio ≈ 32.0
    EXPECT_NEAR(batch_value / single_value, 1.0f, 0.1f)
        << "loss scales with batch size — it is being summed, not averaged. "
        << "Divide the loss by batch size N in cross_entropy_loss.";
}

// ════════════════════════════════════════════════════════════════════════════
// 3.  Optimizer step actually modifies parameter values
//
// If SGD::step() is a no-op or updates a copy rather than the stored tensor,
// parameters never change.  The network would produce the same output every
// forward pass, loss would oscillate around the initial value, and no amount
// of training would help.  This would also cause the single-batch overfit
// to fail — but only if the bug was introduced after that test was written.
// ════════════════════════════════════════════════════════════════════════════

TEST(MnistTrainingDebug, OptimizerStepActuallyModifiesParameters) {
    global_rng = std::mt19937(42);
    MnistDNN net(784, 10);
    SGD      optimiser(net.parameters(), 0.1f);

    // use random non-zero inputs — all-zero inputs kill ReLU gradients
    // and make weight updates zero, which is correct behaviour not a bug
    std::vector<float> random_pixel_values(784 * 4);
    std::mt19937 data_rng(99);
    std::uniform_real_distribution<float> pixel_dist(0.1f, 1.0f);
    for (float& pixel : random_pixel_values) {
        pixel = pixel_dist(data_rng);
    }

    Tensor inputs  = Tensor::from_data(random_pixel_values, {784, 4});
    Tensor targets = Tensor::from_data({0.f, 1.f, 2.f, 3.f}, {1, 4});

    // snapshot before
    std::vector<float> values_before;
    for (Tensor* param : net.parameters()) {
        for (int64_t row = 0; row < param->shape(0); ++row)
            for (int64_t col = 0; col < param->shape(1); ++col)
                values_before.push_back(param->at(row, col));
    }

    Tensor logits = net.forward(inputs);
    Tensor loss   = cross_entropy_loss(logits, targets);
    optimiser.zero_grad();
    backward(loss);
    optimiser.step();

    // snapshot after
    std::vector<float> values_after;
    for (Tensor* param : net.parameters()) {
        for (int64_t row = 0; row < param->shape(0); ++row)
            for (int64_t col = 0; col < param->shape(1); ++col)
                values_after.push_back(param->at(row, col));
    }

    int64_t changed_count = 0;
    int64_t total_count   = static_cast<int64_t>(values_before.size());

    for (size_t index = 0; index < values_before.size(); ++index)
        if (std::abs(values_after[index] - values_before[index]) > 1e-9f)
            ++changed_count;

    printf("\n  Parameters changed: %lld / %lld  (%.1f%%)\n",
           changed_count, total_count,
           100.f * static_cast<float>(changed_count) / static_cast<float>(total_count));

    EXPECT_GT(changed_count, 0)
        << "no parameters changed — step() may be a no-op";

    float fraction_changed = static_cast<float>(changed_count) / static_cast<float>(total_count);
    EXPECT_GT(fraction_changed, 0.5f)
        << "only " << fraction_changed * 100.f << "% of parameters changed";
}
// ════════════════════════════════════════════════════════════════════════════
// 4.  DataLoader visits every sample exactly once per epoch
//
// If the loader silently skips samples, repeats batches, or resets mid-epoch,
// the network sees a biased subset of the data.  For MNIST this would
// manifest as good accuracy on seen classes and near-random on unseen ones,
// keeping overall accuracy artificially high but plateaued.
// ════════════════════════════════════════════════════════════════════════════

TEST(MnistTrainingDebug, DataLoaderVisitsEverySampleExactlyOnce) {
    std::string path   = mnist_path();
    std::string images = path + "/train-images-idx3-ubyte";
    std::string labels = path + "/train-labels-idx1-ubyte";
    {
        std::ifstream file(images);
        if (!file.good()) GTEST_SKIP() << "MNIST files not found at " << path;
    }

    MNISTDataset dataset(images, labels);
    const size_t BATCH_SIZE      = 256;
    const size_t EXPECTED_TOTAL  = 60000;

    DataLoader loader(dataset, BATCH_SIZE, /*shuffle=*/false);

    size_t total_samples_seen = 0;
    int    batch_count        = 0;

    // class distribution — should be roughly uniform for MNIST
    std::vector<int64_t> class_counts(10, 0);

    while (loader.has_next()) {
        auto [inputs, targets_onehot] = loader.next_batch();
        Tensor targets = onehot_to_indices(targets_onehot);

        int64_t batch_n = targets.shape(1);
        total_samples_seen += static_cast<size_t>(batch_n);
        ++batch_count;

        for (int64_t sample = 0; sample < batch_n; ++sample) {
            int64_t cls = static_cast<int64_t>(targets.at(0, sample));
            if (cls >= 0 && cls < 10) ++class_counts[static_cast<size_t>(cls)];
        }
    }

    printf("\n  Batches visited:  %d\n",   batch_count);
    printf("  Total samples:    %zu  (expected %zu)\n",
           total_samples_seen, EXPECTED_TOTAL);
    printf("  Class distribution:\n");
    for (int cls = 0; cls < 10; ++cls) {
        printf("    class %d: %lld samples  (%.1f%%)\n",
               cls, class_counts[cls],
               100.f * static_cast<float>(class_counts[cls])
                     / static_cast<float>(total_samples_seen));
    }

    EXPECT_EQ(total_samples_seen, EXPECTED_TOTAL)
        << "loader visited " << total_samples_seen
        << " samples but dataset has " << EXPECTED_TOTAL;

    // each class should have roughly 6000 samples (10% each) — flag if any
    // class is severely under or over-represented
    for (int cls = 0; cls < 10; ++cls) {
        float fraction = static_cast<float>(class_counts[cls])
                       / static_cast<float>(total_samples_seen);
        EXPECT_GT(fraction, 0.08f)
            << "class " << cls << " appears in only "
            << fraction * 100.f << "% of samples — possible loader skip bug";
        EXPECT_LT(fraction, 0.12f)
            << "class " << cls << " appears in "
            << fraction * 100.f << "% of samples — possible loader repeat bug";
    }
}




TEST(MnistSmoke, TrainsAndLossDecreases) {

    std::string path = mnist_path();

    std::string train_images = path + "/train-images-idx3-ubyte";
    std::string train_labels = path + "/train-labels-idx1-ubyte";
    {
        std::ifstream f(train_images);
        if (!f.good()) {
            GTEST_SKIP() << "MNIST files not found at " << path
                         << " — set MNIST_PATH env var to enable this test";
        }
    }

    // ---- dataset + loader ----
    MNISTDataset train_set(train_images, train_labels);

    const size_t BATCH_SIZE = 16;
    const int    N_EPOCHS   = 3;
    const float  LR         = 0.01f;

    DataLoader loader(train_set, BATCH_SIZE, /*shuffle=*/true);

    // ---- network + optimiser ----
    global_rng = std::mt19937(42);
    MnistDNN net(train_set.input_size(), train_set.num_classes());
    SGD      optim(net.parameters(), LR);

    printf("\n  Epoch   Batches   Avg Loss   Accuracy\n");
    printf("  ------  -------   --------   --------\n");

    float first_epoch_loss = 0.f;
    float last_epoch_loss  = 0.f;
    float last_epoch_acc   = 0.f;

    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {

        loader.reset();

        float total_loss = 0.f;
        int   n_batches  = 0;

        while (loader.has_next()) {

            printf("\r  %5d   %7d   ...", epoch + 1, n_batches);
            fflush(stdout);

            auto [inputs, targets_onehot] = loader.next_batch();
            Tensor targets = onehot_to_indices(targets_onehot);

            Tensor logits = net.forward(inputs);
            Tensor loss   = cross_entropy_loss(logits, targets);

            total_loss += loss.at(0, 0);
            ++n_batches;

            optim.zero_grad();
            backward(loss);
            optim.step();
        }

        float avg_loss = total_loss / (float)n_batches;
        float acc      = evaluate_accuracy(net, loader);

        printf("\r  %5d   %7d   %.6f   %6.2f%%\n",
               epoch + 1, n_batches, avg_loss, acc * 100.f);

        if (epoch == 0)            first_epoch_loss = avg_loss;
        if (epoch == N_EPOCHS - 1) { last_epoch_loss = avg_loss; last_epoch_acc = acc; }
    }

    // ---- loss assertions ----
    EXPECT_LT(last_epoch_loss, first_epoch_loss)
        << "loss should decrease over " << N_EPOCHS << " epochs";
    EXPECT_LT(last_epoch_loss, 2.0f)
        << "loss after " << N_EPOCHS << " epochs should be below 2.0";
    EXPECT_FALSE(std::isnan(last_epoch_loss)) << "loss is NaN";
    EXPECT_FALSE(std::isinf(last_epoch_loss)) << "loss is inf";

    // ---- accuracy assertions ----
    EXPECT_GT(last_epoch_acc, 0.80f)
        << "training accuracy should exceed 80% after " << N_EPOCHS << " epochs";
    EXPECT_FALSE(std::isnan(last_epoch_acc)) << "accuracy is NaN";

    printf("\n  Final avg loss: %.6f  |  Final accuracy: %.2f%%\n",
           last_epoch_loss, last_epoch_acc * 100.f);
}