// tests/profiling/profile_mnist.cpp
//
// Standalone profiling harness — no GTest overhead.
// Runs N batches of the MNIST training loop and exits.
//
// Configuration (env vars or CLI args):
//   MNIST_PATH   path to MNIST binary files   (default: data/MNIST)
//   N_BATCHES    number of batches to run      (default: 200)
//   BATCH_SIZE   samples per batch             (default: 64)
//   LR           SGD learning rate             (default: 0.01)
//
// CLI positional overrides:
//   ./build/profile_mnist [n_batches] [batch_size]
//
// Typical perf usage:
//   cmake --build build --target profile_mnist
//   perf record -F 99 -g ./build/profile_mnist 500
//   perf script | stackcollapse-perf.pl | flamegraph.pl > profile.svg
//
// Or with perf stat for hardware counters:
//   perf stat -e cycles,instructions,cache-misses,cache-references \
//             ./build/profile_mnist 500

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <chrono>

#include "dataset/mnist_dataset.h"
#include "dataset/dataloader.h"
#include "networks/mnist_dnn.h"
#include "loss_functions/cross_entropy.h"
#include "ops/ops.h"
#include "optim.h"

// ----------------------------------------------------------------
// helpers
// ----------------------------------------------------------------

static std::string env_str(const char* key, const char* fallback) {
    const char* v = std::getenv(key);
    return v ? std::string(v) : std::string(fallback);
}

static int env_int(const char* key, int fallback) {
    const char* v = std::getenv(key);
    return v ? std::atoi(v) : fallback;
}

static float env_float(const char* key, float fallback) {
    const char* v = std::getenv(key);
    return v ? (float)std::atof(v) : fallback;
}

// Convert one-hot [10, N] → class-index [1, N] for cross-entropy
static Tensor onehot_to_indices(const Tensor& onehot) {
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

// ----------------------------------------------------------------
// main
// ----------------------------------------------------------------

int main(int argc, char* argv[]) {

    // ---- config ------------------------------------------------
    std::string mnist_path = env_str("MNIST_PATH", "data/MNIST");
    int   n_batches  = env_int  ("N_BATCHES",  200);
    int   batch_size = env_int  ("BATCH_SIZE",  64);
    float lr         = env_float("LR",         0.01f);

    // CLI positional overrides
    if (argc >= 2) n_batches  = std::atoi(argv[1]);
    if (argc >= 3) batch_size = std::atoi(argv[2]);

    // ---- banner ------------------------------------------------
    printf("=== MNIST Profiling Harness ===\n");
    printf("  MNIST path  : %s\n",   mnist_path.c_str());
    printf("  n_batches   : %d\n",   n_batches);
    printf("  batch_size  : %d\n",   batch_size);
    printf("  lr          : %.4f\n", lr);
    printf("\n");

    // ---- dataset -----------------------------------------------
    std::string img_path = mnist_path + "/train-images-idx3-ubyte";
    std::string lbl_path = mnist_path + "/train-labels-idx1-ubyte";

    {
        FILE* f = std::fopen(img_path.c_str(), "rb");
        if (!f) {
            fprintf(stderr,
                "ERROR: MNIST images not found at '%s'\n"
                "       Set MNIST_PATH env var to point at your MNIST files\n",
                img_path.c_str());
            return 1;
        }
        std::fclose(f);
    }

    MNISTDataset train_set(img_path, lbl_path);
    DataLoader   loader(train_set, (size_t)batch_size, /*shuffle=*/true);

    // ---- model + optimiser ------------------------------------
    global_rng = std::mt19937(42);
    MnistDNN net;
    SGD      optim(net.parameters(), lr);

    // ---- warm-up pass (excluded from timing) ------------------
    // Ensures allocations and any lazy-init are done before we
    // start profiling so the profile reflects steady-state cost.
    {
        printf("Warming up (1 batch)... ");
        fflush(stdout);

        loader.reset();
        auto [inputs, targets_onehot] = loader.next_batch();
        Tensor targets = onehot_to_indices(targets_onehot);
        Tensor logits  = net.forward(inputs);
        Tensor loss    = cross_entropy_loss(logits, targets);
        optim.zero_grad();
        backward(loss);
        optim.step();

        printf("done.\n\n");
    }

    // ---- profiling loop ----------------------------------------
    printf("Running %d batches (hot loop — attach profiler now if using external tool)...\n\n",
           n_batches);

    loader.reset();

    float   total_loss   = 0.f;
    int     batches_done = 0;
    int     resets       = 0;

    using Clock = std::chrono::steady_clock;
    auto t_start = Clock::now();

    while (batches_done < n_batches) {

        // Wrap around the dataset if we exhaust it
        if (!loader.has_next()) {
            loader.reset();
            ++resets;
        }

        auto [inputs, targets_onehot] = loader.next_batch();
        Tensor targets = onehot_to_indices(targets_onehot);

        // ---- forward ----
        Tensor logits = net.forward(inputs);

        // ---- loss ----
        Tensor loss = cross_entropy_loss(logits, targets);
        total_loss += loss.at({0, 0});

        // ---- backward + update ----
        optim.zero_grad();
        backward(loss);
        optim.step();

        ++batches_done;

        // Progress every 50 batches
        if (batches_done % 50 == 0) {
            float avg = total_loss / (float)batches_done;
            printf("  [%4d / %4d]  avg loss: %.5f\n",
                   batches_done, n_batches, avg);
            fflush(stdout);
        }
    }

    auto t_end = Clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // ---- summary -----------------------------------------------
    float  avg_loss        = total_loss / (float)batches_done;
    double ms_per_batch    = elapsed_ms / (double)batches_done;
    double samples_per_sec = (1000.0 / ms_per_batch) * batch_size;

    printf("\n=== Results ===\n");
    printf("  Batches run     : %d  (dataset wrapped %d time(s))\n",
           batches_done, resets);
    printf("  Avg loss        : %.5f\n", avg_loss);
    printf("  Total time      : %.1f ms\n", elapsed_ms);
    printf("  Time / batch    : %.2f ms\n", ms_per_batch);
    printf("  Throughput      : %.0f samples/sec\n", samples_per_sec);

    return 0;
}