// tests/profiling/bench_mnist.cpp
//
// Full MNIST training-loop benchmark.  No GTest overhead — just raw throughput.
//
// Configuration (env vars / CLI positional args):
//   N_BATCHES   number of batches to run   (default: 200)
//   BATCH_SIZE  samples per batch          (default: 64)
//   LR          SGD learning rate          (default: 0.1)
//   MODEL       mlp or cnn                 (default: mlp)
//
//   ./build/bench_mnist [n_batches] [batch_size] [--model mlp|cnn]
//
// Profiler usage:
//   cmake --build build --target bench_mnist
//   perf record -F 99 -g ./build/bench_mnist 500
//   perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
//
//   perf stat -e cycles,instructions,cache-misses \
//             ./build/bench_mnist 500

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "autograd.h"
#include "dataset/dataloader.h"
#include "dataset/mnist_dataset.h"
#include "loss_functions/cross_entropy.h"
#include "networks/cnn.h"
#include "networks/mlp.h"
#include "ops/activations/relu.h"
#include "ops/reshape.h"
#include "optim/sgd.h"

// ─────────────────────────────────────────────
// helpers
// ─────────────────────────────────────────────

static int env_int(const char* key, int fallback) {
    const char* v = std::getenv(key);
    return v ? std::atoi(v) : fallback;
}

static float env_float(const char* key, float fallback) {
    const char* v = std::getenv(key);
    return v ? static_cast<float>(std::atof(v)) : fallback;
}

static std::string env_str(const char* key, const std::string& fallback) {
    const char* v = std::getenv(key);
    return v ? std::string(v) : fallback;
}

static std::string resolve_mnist(const std::string& rel) {
    for (const std::string& base : {"", "../", "/workspace/"}) {
        std::string p = base + rel;
        if (std::filesystem::exists(p)) return p;
    }
    return rel;
}

static Shape make_shape_4d(size_t d0, size_t d1, size_t d2, size_t d3) {
    Shape s{};
    s[0] = d0;
    s[1] = d1;
    s[2] = d2;
    s[3] = d3;
    return s;
}

static std::string parse_model_arg(int argc, char* argv[], const std::string& fallback) {
    std::string model = fallback;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model = argv[++i];
            continue;
        }
        if (arg.rfind("--model=", 0) == 0) {
            model = arg.substr(8);
            continue;
        }
    }
    return model;
}

// ─────────────────────────────────────────────
// main
// ─────────────────────────────────────────────

int main(int argc, char* argv[]) {
    int   n_batches  = env_int  ("N_BATCHES",  200);
    int   batch_size = env_int  ("BATCH_SIZE",  64);
    float lr         = env_float("LR",          0.1f);
    std::string model_name = env_str("MODEL", "mlp");

    if (argc >= 2) n_batches  = std::atoi(argv[1]);
    if (argc >= 3) batch_size = std::atoi(argv[2]);
    model_name = parse_model_arg(argc, argv, model_name);

    if (model_name != "mlp" && model_name != "cnn") {
        std::fprintf(stderr, "ERROR: --model must be 'mlp' or 'cnn' (got '%s')\n", model_name.c_str());
        return 1;
    }

    const std::string img_path =
        resolve_mnist("data/MNIST/train-images-idx3-ubyte");
    const std::string lbl_path =
        resolve_mnist("data/MNIST/train-labels-idx1-ubyte");

    if (!std::filesystem::exists(img_path)) {
        std::fprintf(stderr,
            "ERROR: MNIST images not found at '%s'\n"
            "       Place the MNIST binary files under data/MNIST/ "
            "or set N_BATCHES=0 to skip.\n",
            img_path.c_str());
        return 1;
    }

    std::printf("=== bench_mnist ===\n");
    std::printf("  MNIST path  : %s\n", img_path.c_str());
    std::printf("  n_batches   : %d\n", n_batches);
    std::printf("  batch_size  : %d\n", batch_size);
    std::printf("  lr          : %.4f\n\n", lr);
    std::printf("  model       : %s\n\n", model_name.c_str());

    MNISTDataset dataset(img_path, lbl_path);
    DataLoader   loader(dataset, static_cast<size_t>(batch_size),
                        /*shuffle=*/true, /*seed=*/42u);

    // Infer dimensions from the first real batch.
    auto [probe_x, probe_y]      = loader.next_batch();
    const size_t input_features  = probe_x.shape_at(1);
    const size_t num_classes     = probe_y.shape_at(1);
    const size_t image_rows      = dataset.image_rows();
    const size_t image_cols      = dataset.image_cols();
    loader.reset();

    MLP mlp_model(input_features, {128, 64}, relu, num_classes);
    CNN cnn_model(
        /*input_channels=*/1,
        image_rows,
        image_cols,
        /*conv_out_channels=*/8,
        /*kernel_h=*/3,
        /*kernel_w=*/3,
        relu,
        num_classes,
        /*stride_h=*/1,
        /*stride_w=*/1,
        /*padding_h=*/1,
        /*padding_w=*/1
    );

    std::vector<Tensor*> params =
        (model_name == "cnn") ? cnn_model.parameters() : mlp_model.parameters();
    SGD optim(params, lr);

    // ── warm-up (excluded from timing) ───────────────────────
    std::printf("Warming up (1 batch)... ");
    std::fflush(stdout);
    {
        auto [x, y] = loader.next_batch();
        Tensor logits = Tensor::zeros(probe_y.shape, probe_y.ndim);
        if (model_name == "cnn") {
            const size_t b = x.shape_at(0);
            Tensor x_nchw = reshape(x, make_shape_4d(b, 1, image_rows, image_cols), 4);
            logits = cnn_model.forward(x_nchw);
        } else {
            logits = mlp_model.forward(x);
        }
        Tensor loss   = cross_entropy(logits, y);
        backward(loss);
        optim.step();
        optim.zero_grad();
    }
    loader.reset();
    std::printf("done.\n\n");

    // ── hot loop ──────────────────────────────────────────────
    std::printf("Running %d batches "
                "(attach profiler now if using an external tool)...\n\n",
                n_batches);

    float  total_loss   = 0.f;
    int    batches_done = 0;
    int    resets       = 0;

    using Clock = std::chrono::steady_clock;
    const auto t_start = Clock::now();

    while (batches_done < n_batches) {
        if (!loader.has_next()) { loader.reset(); ++resets; }

        auto [x, y]   = loader.next_batch();
        Tensor logits = Tensor::zeros(probe_y.shape, probe_y.ndim);
        if (model_name == "cnn") {
            const size_t b = x.shape_at(0);
            Tensor x_nchw = reshape(x, make_shape_4d(b, 1, image_rows, image_cols), 4);
            logits = cnn_model.forward(x_nchw);
        } else {
            logits = mlp_model.forward(x);
        }
        Tensor loss   = cross_entropy(logits, y);

        total_loss += loss(0);

        backward(loss);
        optim.step();
        optim.zero_grad();

        ++batches_done;

        if (batches_done % 50 == 0) {
            std::printf("  [%4d / %4d]  avg loss: %.5f\n",
                        batches_done, n_batches,
                        total_loss / static_cast<float>(batches_done));
            std::fflush(stdout);
        }
    }

    const auto t_end = Clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // ── summary ───────────────────────────────────────────────
    const double ms_per_batch    = elapsed_ms / batches_done;
    const double samples_per_sec = (1000.0 / ms_per_batch) * batch_size;

    std::printf("\n=== Results ===\n");
    std::printf("  Batches run     : %d  (dataset wrapped %d time(s))\n",
                batches_done, resets);
    std::printf("  Avg loss        : %.5f\n",
                total_loss / static_cast<float>(batches_done));
    std::printf("  Total time      : %.1f ms\n",    elapsed_ms);
    std::printf("  Time / batch    : %.2f ms\n",    ms_per_batch);
    std::printf("  Throughput      : %.0f samples/sec\n", samples_per_sec);

    return 0;
}
