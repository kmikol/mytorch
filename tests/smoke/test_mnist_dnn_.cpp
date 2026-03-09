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
    const char* env = std::getenv("MNIST_PATH");
    return env ? std::string(env) : "data/MNIST";
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
            float   best_val   = logits.at({0, n});
            for (int64_t c = 1; c < 10; c++) {
                float v = logits.at({c, n});
                if (v > best_val) { best_val = v; pred_class = c; }
            }

            // true class: argmax over one-hot target for sample n
            int64_t true_class = 0;
            for (int64_t c = 1; c < 10; c++)
                if (targets.at({c, n}) > targets.at({true_class, n}))
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
            if (onehot.at({c, n}) > 0.5f) {
                indices[n] = (float)c;
                break;
            }
        }
    }
    return Tensor::from_data(indices, {1, N});
}

// ================================================================
// smoke test: train for N epochs, print loss, assert it decreases
// ================================================================

TEST(MnistSmoke, TrainsAndLossDecreases) {

    std::string path = mnist_path();

    // skip gracefully if MNIST files aren't present
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

    const size_t BATCH_SIZE = 64;
    const int    N_EPOCHS   = 3;
    const float  LR         = 0.01f;

    DataLoader loader(train_set, BATCH_SIZE, /*shuffle=*/true);

    // ---- network + optimiser ----
    global_rng = std::mt19937(42);
    MnistDNN net;
    SGD      optim(net.parameters(), LR);

    printf("\n  Epoch   Batches   Avg Loss\n");
    printf("  ------  -------   --------\n");

    float first_epoch_loss = 0.f;
    float last_epoch_loss  = 0.f;

    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {

        printf("loader reset");

        loader.reset();

        float total_loss  = 0.f;
        int   n_batches   = 0;

        while (loader.has_next()) {

            printf("\r  %5d   %7d   ...", epoch + 1, n_batches);
            fflush(stdout);

            auto [inputs, targets_onehot] = loader.next_batch();

            // cross_entropy_loss wants class indices [1, N]
            Tensor targets = onehot_to_indices(targets_onehot);

            Tensor logits = net.forward(inputs);
            Tensor loss   = cross_entropy_loss(logits, targets);

            total_loss += loss.at({0, 0});
            ++n_batches;

            optim.zero_grad();
            backward(loss);
            optim.step();
        }

        float avg_loss = total_loss / (float)n_batches;

        printf("  %5d   %7d   %.6f\n", epoch + 1, n_batches, avg_loss);

        if (epoch == 0)           first_epoch_loss = avg_loss;
        if (epoch == N_EPOCHS - 1) last_epoch_loss  = avg_loss;
    }

    // ---- assertions ----

    // loss must have decreased over training
    EXPECT_LT(last_epoch_loss, first_epoch_loss)
        << "loss should decrease over " << N_EPOCHS << " epochs";

    // loss should be in a reasonable range — random init gives ~log(10)≈2.3
    // after a few epochs it should be well below that
    EXPECT_LT(last_epoch_loss, 2.0f)
        << "loss after " << N_EPOCHS << " epochs should be below 2.0";

    // loss should not have exploded
    EXPECT_FALSE(std::isnan(last_epoch_loss))  << "loss is NaN";
    EXPECT_FALSE(std::isinf(last_epoch_loss))  << "loss is inf";

    printf("\n  Final avg loss: %.6f\n", last_epoch_loss);
}

// ----------------------------------------------------------------
// accuracy smoke test: after a few epochs, accuracy should beat
// random chance (10%) by a meaningful margin
// ----------------------------------------------------------------
TEST(MnistSmoke, AccuracyBeatRandom) {

    std::string path = mnist_path();

    std::string train_images = path + "/train-images-idx3-ubyte";
    std::string train_labels = path + "/train-labels-idx1-ubyte";
    {
        std::ifstream f(train_images);
        if (!f.good()) {
            GTEST_SKIP() << "MNIST files not found at " << path;
        }
    }

    MNISTDataset train_set(train_images, train_labels);

    const size_t BATCH_SIZE = 64;
    const int    N_EPOCHS   = 2;
    const float  LR         = 0.01f;

    DataLoader loader(train_set, BATCH_SIZE, true);

    global_rng = std::mt19937(42);
    MnistDNN net;
    SGD      optim(net.parameters(), LR);

    // train
    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        loader.reset();
        while (loader.has_next()) {
            auto [inputs, targets_onehot] = loader.next_batch();
            Tensor targets = onehot_to_indices(targets_onehot);
            Tensor logits  = net.forward(inputs);
            Tensor loss    = cross_entropy_loss(logits, targets);
            optim.zero_grad();
            backward(loss);
            optim.step();
        }
    }

    // evaluate on training set
    // (we don't have a separate test set loader here — smoke test only)
    float acc = evaluate_accuracy(net, loader);

    printf("\n  Training accuracy after %d epochs: %.1f%%\n",
           N_EPOCHS, acc * 100.f);

    // random chance is 10% — we should comfortably beat it after 2 epochs
    EXPECT_GT(acc, 0.50f)
        << "accuracy should exceed 50% after " << N_EPOCHS << " epochs";
}

// ----------------------------------------------------------------
// minimal sanity: single forward pass produces the right output shape
// no training, no files needed
// ----------------------------------------------------------------
TEST(MnistSmoke, ForwardOutputShape) {
    global_rng = std::mt19937(42);
    MnistDNN net;

    // fake batch of 8 images
    Tensor x = Tensor::zeros({784, 8});
    Tensor out = net.forward(x);

    // output should be [10, 8] — 10 class logits per sample
    EXPECT_EQ(out.shape(0), 10);
    EXPECT_EQ(out.shape(1), 8);
}

TEST(MnistSmoke, ForwardOutputIsFinite) {
    global_rng = std::mt19937(42);
    MnistDNN net;

    Tensor x   = Tensor::zeros({784, 4});
    Tensor out = net.forward(x);

    for (int64_t c = 0; c < 10; c++)
        for (int64_t n = 0; n < 4; n++) {
            EXPECT_FALSE(std::isnan(out.at({c, n}))) << "nan at [" << c << "," << n << "]";
            EXPECT_FALSE(std::isinf(out.at({c, n}))) << "inf at [" << c << "," << n << "]";
        }
}

TEST(MnistSmoke, ParametersReceiveGradients) {
    // single forward+backward — all parameters should have gradients
    global_rng = std::mt19937(42);
    MnistDNN net;
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