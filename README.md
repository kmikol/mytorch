# mytorch

A from-scratch C++ implementation of the core machinery behind PyTorch — tensors, autograd,
layers, an optimizer, and a full training pipeline — built to develop a ground-level
understanding of how deep learning frameworks actually work.

Trains a CNN on MNIST end-to-end with no external ML dependencies.

---

## What is implemented

**Autograd engine** — dynamic computation graph with reverse-mode automatic differentiation.
Every operation registers its backward function at forward time; `backward()` walks the graph
and accumulates gradients.

**Layers**
- `Conv2d` — implemented via im2col for efficient matrix multiplication, configurable kernel,
  stride, and padding
- `Linear` — fully connected layer with weight and bias gradients

**Activations and ops** — `ReLU` (mask-gated backward), `Reshape`, `Sigmoid`, matrix multiply,
elementwise add/mul, and more in `src/ops/`

**Loss** — `CrossEntropy` as fused log-softmax + NLL, giving a clean closed-form gradient at
the logit layer

**Optimizer** — SGD with `step()` and `zero_grad()`

**Models** — `CNN` (Conv2d → ReLU → Reshape → Linear) and `MLP` both included

**Data pipeline** — MNIST binary file loader with one-hot encoding, shuffling, and batching

**Parallelism** — OpenMP used across compute-heavy ops

---

## Requirements

- C++20 compiler (GCC 11+ or Clang 13+)
- CMake 3.20+
- Ninja
- GTest
- OpenMP
- gcovr (for coverage reports)
- gprof (for profiling, optional)
- BLAS (optional — enables reference comparison in op benchmarks)

A `.devcontainer` configuration is included for a fully reproducible environment in VS Code or
GitHub Codespaces. A `dockerfile` is also provided for standalone container builds.

---

## Getting started

```bash
# Clone and enter
git clone https://github.com/kmikol/mytorch.git
cd mytorch

# Train (5 epochs, debug build)
make run_main

# Train with options
make run_main mode=opt epochs=20
```

---

## Commands

```bash
# Tests
make unit_tests                          # unit tests
make smoke_tests                         # smoke tests
make test_datasets                       # dataset tests
make all_tests                           # all three

# Filter to a specific test
make unit_tests filter=MatMulOpForwardTest.*

# Coverage (HTML report)
make coverage
# open build/coverage_html/index.html

# Benchmarks
make bench op=conv2d mode=forward        # single op benchmark
make bench op=matmul mode=backward size=1024 iters=200
make bench_mnist                         # full training throughput
make bench_mnist bench_model=cnn batches=100

# Profiling
make profile_gprof profile_epochs=3
make profile_gprof_top

# Clean
make clean
```

Available ops for `make bench`: `matmul`, `mul`, `add`, `relu`, `sigmoid`, `cross_entropy`,
`linear`, `conv2d`, `im2col`, `reshape`

---

## Project structure

```
src/
  tensor/          # Tensor class and autograd graph
  layers/          # Conv2d, Linear
  ops/             # Activations, matmul, reshape, and other ops
  loss_functions/  # CrossEntropy
  optim/           # SGD
  networks/        # CNN, MLP
  dataset/         # MNIST loader and DataLoader
  utils/           # Metrics
tests/
  unit/            # Unit tests (GTest)
  smoke/           # Smoke tests
  datasets/        # Dataset tests
  profiling/       # bench_ops, bench_mnist
data/
  MNIST/           # Binary MNIST files (not committed)
```

---

## Implementation notes

**im2col for Conv2d** — rather than implementing the convolution loop directly, the forward
pass uses im2col to reshape the input into a matrix so the computation reduces to a single
matrix multiply. The backward pass uses the transpose (col2im) to propagate gradients.

**Fused CrossEntropy** — log-softmax and NLL are fused rather than chained as separate
operations. This avoids numerical instability from computing `exp()` without max-subtraction
and gives a cleaner gradient: `(p - y) / batch_size` at the logit layer.

**Gradient accumulation** — gradients accumulate across backward calls until `zero_grad()` is
called explicitly, matching PyTorch's behaviour and making the interaction visible rather than
hidden.
