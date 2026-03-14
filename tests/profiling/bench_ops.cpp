// tests/profiling/bench_ops.cpp
//
// Microbenchmark for individual ops in forward or backward mode.
//
// Usage:
//   bench_ops --op <name> --mode <forward|backward> [--size N] [--iters N] [--warmup N]
//
// Ops:   matmul  mul  add  relu  sigmoid  cross_entropy  linear
// Modes: forward  backward
//
// --size N   primary dimension (default 512).
//            matmul      : [N×N] @ [N×N]
//            mul/add     : [N×N] element-wise
//            relu/sigmoid: [N×N] element-wise
//            cross_entropy: [N×10] batch, 10 classes
//            linear      : [N×784] input, Linear(784,256)
//
// Examples (via cmake convenience targets):
//   cmake --build build --target bench_matmul_forward
//   cmake --build build --target bench_relu_backward
//
// Or directly:
//   ./build/bench_ops --op matmul --mode forward --size 512 --iters 50
//   ./build/bench_ops --op sigmoid --mode backward --size 1024

#include <algorithm>
#ifdef HAVE_BLAS
#include <cblas.h>
#endif
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "autograd.h"
#include "layers/linear.h"
#include "loss_functions/cross_entropy.h"
#include "ops/activations/relu.h"
#include "ops/activations/sigmoid.h"
#include "ops/add.h"
#include "ops/matmul.h"
#include "ops/mul.h"

// ─────────────────────────────────────────────
// Timing
// ─────────────────────────────────────────────

using Clock = std::chrono::steady_clock;

struct BenchResult {
    double mean_ms;
    double min_ms;
    double median_ms;
    size_t iters;
};

// Runs `fn` for `warmup` iterations (excluded from stats), then `iters`
// timed iterations.  Returns per-iteration statistics in milliseconds.
static BenchResult run_bench(size_t iters, size_t warmup,
                             const std::function<void()>& fn) {
    for (size_t i = 0; i < warmup; ++i) fn();

    std::vector<double> times;
    times.reserve(iters);

    for (size_t i = 0; i < iters; ++i) {
        const auto t0 = Clock::now();
        fn();
        const auto t1 = Clock::now();
        times.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::sort(times.begin(), times.end());
    const double total = [&] {
        double s = 0; for (double t : times) s += t; return s;
    }();

    return {
        total / static_cast<double>(iters),
        times.front(),
        times[iters / 2],
        iters,
    };
}

// ─────────────────────────────────────────────
// Tensor helpers
// ─────────────────────────────────────────────

static Shape make_shape(std::initializer_list<size_t> dims) {
    Shape s{};
    size_t i = 0;
    for (size_t d : dims) s[i++] = d;
    return s;
}

static Tensor rand_tensor(std::initializer_list<size_t> dims,
                          bool requires_grad = false) {
    Shape s = make_shape(dims);
    const size_t nd = dims.size();
    Tensor t(s, nd, requires_grad);
    size_t n = 1;
    for (size_t d : dims) n *= d;
    // Fill with deterministic pseudo-random values in (-1, 1).
    for (size_t i = 0; i < n; ++i)
        t.storage->data[i] = static_cast<float>((i * 6364136223846793005ULL + 1) % 1000) / 500.f - 1.f;
    return t;
}

// Softmax along axis 1 — ensures cross_entropy receives valid probabilities.
static Tensor softmax_rows(const Tensor& x) {
    const size_t N = x.shape_at(0);
    const size_t C = x.shape_at(1);
    Tensor out = Tensor::zeros(x.shape, x.ndim);
    for (size_t i = 0; i < N; ++i) {
        float mx = x(i, 0);
        for (size_t j = 1; j < C; ++j) mx = std::max(mx, x(i, j));
        float sum = 0.f;
        for (size_t j = 0; j < C; ++j) {
            out(i, j) = std::exp(x(i, j) - mx);
            sum += out(i, j);
        }
        for (size_t j = 0; j < C; ++j) out(i, j) /= sum;
    }
    return out;
}

// One-hot tensor: each row has a 1 at a fixed class.
static Tensor onehot(size_t N, size_t C) {
    Tensor t = Tensor::zeros(make_shape({N, C}), 2);
    for (size_t i = 0; i < N; ++i)
        t(i, i % C) = 1.f;
    return t;
}

// ─────────────────────────────────────────────
// Print
// ─────────────────────────────────────────────

static void print_header(const char* op, const char* mode, size_t size,
                         size_t iters, size_t warmup) {
    std::printf("\n=== bench_ops: %s %s ===\n", op, mode);
    std::printf("  Size        : %zu\n", size);
    std::printf("  Iterations  : %zu  (+ %zu warmup)\n", iters, warmup);
}

static void print_result(const BenchResult& r) {
    std::printf("  ─────────────────────────────\n");
    std::printf("  Mean        : %8.3f ms\n", r.mean_ms);
    std::printf("  Min         : %8.3f ms\n", r.min_ms);
    std::printf("  Median      : %8.3f ms\n", r.median_ms);
}

static void print_gflops(const BenchResult& r, double flops) {
    const double gflops = flops / (r.mean_ms * 1e-3) / 1e9;
    std::printf("  GFLOPs      : %8.2f  (at mean)\n", gflops);
}

// ─────────────────────────────────────────────
// Per-op benchmarks
// ─────────────────────────────────────────────

static void bench_matmul(const std::string& mode, size_t N,
                         size_t iters, size_t warmup) {
    auto A = rand_tensor({N, N});
    auto B = rand_tensor({N, N});
    auto G = rand_tensor({N, N});

    print_header("matmul", mode.c_str(), N, iters, warmup);

    BenchResult r;
    if (mode == "forward") {
        r = run_bench(iters, warmup, [&] { MatMulOp::forward(A, B); });
    } else {
        r = run_bench(iters, warmup, [&] { MatMulOp::backward(G, A, B); });
    }

    print_result(r);
    // matmul FLOPs: 2 * N^3 (forward); backward does 2 matmuls → 4 * N^3
    const double flops = (mode == "forward" ? 2.0 : 4.0) * N * N * N;
    print_gflops(r, flops);

    // ── BLAS reference (forward only) ─────────────────────────
#ifdef HAVE_BLAS
    if (mode == "forward") {
        // cblas_sgemm: C = alpha * A @ B + beta * C
        std::vector<float> ca(N * N), cb(N * N), cc(N * N, 0.f);
        for (size_t i = 0; i < N * N; ++i) {
            ca[i] = A.storage->data[i];
            cb[i] = B.storage->data[i];
        }
        BenchResult rb = run_bench(iters, warmup, [&] {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(N), static_cast<int>(N), static_cast<int>(N),
                        1.f, ca.data(), static_cast<int>(N),
                             cb.data(), static_cast<int>(N),
                        0.f, cc.data(), static_cast<int>(N));
        });
        const double blas_gflops = flops / (rb.mean_ms * 1e-3) / 1e9;
        const double our_gflops  = flops / (r.mean_ms  * 1e-3) / 1e9;
        std::printf("  BLAS mean   : %8.3f ms  (%.2f GFLOP/s)\n",
                    rb.mean_ms, blas_gflops);
        std::printf("  vs BLAS     : %8.1f %%\n", 100.0 * our_gflops / blas_gflops);
    }
#endif
    std::printf("  ─────────────────────────────\n");
}

static void bench_mul(const std::string& mode, size_t N,
                      size_t iters, size_t warmup) {
    auto a = rand_tensor({N, N});
    auto b = rand_tensor({N, N});
    auto g = rand_tensor({N, N});

    print_header("mul", mode.c_str(), N, iters, warmup);

    BenchResult r;
    if (mode == "forward") {
        r = run_bench(iters, warmup, [&] { MulOp::forward(a, b); });
    } else {
        r = run_bench(iters, warmup, [&] { MulOp::backward(g, a, b); });
    }

    print_result(r);
    std::printf("  ─────────────────────────────\n");
}

static void bench_add(const std::string& mode, size_t N,
                      size_t iters, size_t warmup) {
    auto a = rand_tensor({N, N});
    auto b = rand_tensor({N, N});
    auto g = rand_tensor({N, N});

    print_header("add", mode.c_str(), N, iters, warmup);

    BenchResult r;
    if (mode == "forward") {
        r = run_bench(iters, warmup, [&] { AddOp::forward(a, b); });
    } else {
        r = run_bench(iters, warmup, [&] { AddOp::backward(g, a, b); });
    }

    print_result(r);
    std::printf("  ─────────────────────────────\n");
}

static void bench_relu(const std::string& mode, size_t N,
                       size_t iters, size_t warmup) {
    auto x = rand_tensor({N, N});
    auto g = rand_tensor({N, N});

    print_header("relu", mode.c_str(), N, iters, warmup);

    BenchResult r;
    if (mode == "forward") {
        r = run_bench(iters, warmup, [&] { ReLUOp::forward(x); });
    } else {
        r = run_bench(iters, warmup, [&] { ReLUOp::backward(g, x); });
    }

    print_result(r);
    std::printf("  ─────────────────────────────\n");
}

static void bench_sigmoid(const std::string& mode, size_t N,
                          size_t iters, size_t warmup) {
    auto x   = rand_tensor({N, N});
    auto out = SigmoidOp::forward(x);   // backward needs σ(x), not x
    auto g   = rand_tensor({N, N});

    print_header("sigmoid", mode.c_str(), N, iters, warmup);

    BenchResult r;
    if (mode == "forward") {
        r = run_bench(iters, warmup, [&] { SigmoidOp::forward(x); });
    } else {
        r = run_bench(iters, warmup, [&] { SigmoidOp::backward(g, out); });
    }

    print_result(r);
    std::printf("  ─────────────────────────────\n");
}

static void bench_cross_entropy(const std::string& mode, size_t N,
                                size_t iters, size_t warmup) {
    constexpr size_t C = 10;
    auto probs   = softmax_rows(rand_tensor({N, C}));
    auto targets = onehot(N, C);
    auto g       = Tensor::ones(make_shape({1}), 1);

    print_header("cross_entropy", mode.c_str(), N, iters, warmup);
    std::printf("  Classes     : %zu\n", C);

    BenchResult r;
    if (mode == "forward") {
        r = run_bench(iters, warmup,
                      [&] { CrossEntropyOp::forward(probs, targets); });
    } else {
        r = run_bench(iters, warmup,
                      [&] { CrossEntropyOp::backward(g, probs, targets); });
    }

    print_result(r);
    std::printf("  ─────────────────────────────\n");
}

static void bench_linear(const std::string& mode, size_t N,
                         size_t iters, size_t warmup) {
    constexpr size_t IN  = 784;
    constexpr size_t OUT = 256;

    Linear layer(IN, OUT);
    auto   x = rand_tensor({N, IN}, /*requires_grad=*/true);

    print_header("linear", mode.c_str(), N, iters, warmup);
    std::printf("  Shape       : [%zu×%zu] → [%zu×%zu]\n", N, IN, N, OUT);

    BenchResult r;
    if (mode == "forward") {
        r = run_bench(iters, warmup, [&] { layer.forward(x); });
    } else {
        r = run_bench(iters, warmup, [&] {
            // Rebuild graph each iteration — forward is part of the backward cost.
            auto y = layer.forward(x);
            backward(y);
            layer.weight.autograd_meta = nullptr;
            layer.bias.autograd_meta   = nullptr;
            x.autograd_meta            = nullptr;
            // Reinstate requires_grad for the next iteration.
            layer.weight.autograd_meta = std::make_shared<AutogradMeta>();
            layer.weight.autograd_meta->requires_grad = true;
            layer.bias.autograd_meta   = std::make_shared<AutogradMeta>();
            layer.bias.autograd_meta->requires_grad   = true;
            x.autograd_meta            = std::make_shared<AutogradMeta>();
            x.autograd_meta->requires_grad            = true;
        });
    }

    print_result(r);
    std::printf("  ─────────────────────────────\n");
}

// ─────────────────────────────────────────────
// Argument parsing
// ─────────────────────────────────────────────

static void print_usage(const char* prog) {
    std::printf(
        "Usage: %s --op <name> --mode <forward|backward>"
        " [--size N] [--iters N] [--warmup N]\n\n"
        "Ops:   matmul  mul  add  relu  sigmoid  cross_entropy  linear\n"
        "Modes: forward  backward\n\n"
        "Defaults: --size 512 --iters 100 --warmup 10\n",
        prog);
}

int main(int argc, char* argv[]) {
    std::string op;
    std::string mode;
    size_t size   = 512;
    size_t iters  = 100;
    size_t warmup = 10;

    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--op")     == 0 && i+1 < argc) op     = argv[++i];
        else if (std::strcmp(argv[i], "--mode")   == 0 && i+1 < argc) mode   = argv[++i];
        else if (std::strcmp(argv[i], "--size")   == 0 && i+1 < argc) size   = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters")  == 0 && i+1 < argc) iters  = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i+1 < argc) warmup = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--help")   == 0) { print_usage(argv[0]); return 0; }
    }

    if (op.empty() || mode.empty()) {
        std::fprintf(stderr, "ERROR: --op and --mode are required.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    if (mode != "forward" && mode != "backward") {
        std::fprintf(stderr, "ERROR: --mode must be 'forward' or 'backward'.\n");
        return 1;
    }

    if      (op == "matmul")        bench_matmul       (mode, size, iters, warmup);
    else if (op == "mul")           bench_mul          (mode, size, iters, warmup);
    else if (op == "add")           bench_add          (mode, size, iters, warmup);
    else if (op == "relu")          bench_relu         (mode, size, iters, warmup);
    else if (op == "sigmoid")       bench_sigmoid      (mode, size, iters, warmup);
    else if (op == "cross_entropy") bench_cross_entropy(mode, size, iters, warmup);
    else if (op == "linear")        bench_linear       (mode, size, iters, warmup);
    else {
        std::fprintf(stderr,
            "ERROR: unknown op '%s'.\n"
            "       Valid ops: matmul mul add relu sigmoid cross_entropy linear\n",
            op.c_str());
        return 1;
    }

    return 0;
}
