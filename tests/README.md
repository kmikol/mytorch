# Tests

## Directory Structure

```
tests/
├── unit/               # Isolated unit tests — one file per source module
│   ├── tensor/         # Storage and Tensor primitives
│   ├── ops/            # Element-wise and matrix ops (mul, add, matmul, relu, sigmoid)
│   ├── layers/         # Learnable layers (Linear)
│   ├── loss_functions/ # Loss functions (CrossEntropy)
│   ├── optim/          # Optimisers (SGD)
│   ├── utils/          # Utilities (metrics)
│   └── dataset/        # Dataset and DataLoader
├── smoke/              # End-to-end integration tests (full training runs)
├── profiling/          # Performance benchmarks (not part of CI)
└── datasets/           # Dataset-specific tests (requires data files)
```

## Building and Running

```bash
cmake --build /workspace/build --target unit_tests -j$(nproc)
/workspace/build/unit_tests

# Run a single test suite
/workspace/build/unit_tests --gtest_filter="MulOpForwardTest.*"
```

---

## Test Suite Naming Convention

Every op or module uses the same four-tier structure:

| Suite name              | What it tests                                                    |
|-------------------------|------------------------------------------------------------------|
| `<Name>ForwardTest`     | `<Name>Op::forward()` — pure computation, no autograd           |
| `<Name>BackwardTest`    | `<Name>Op::backward()` — gradient formulas in isolation          |
| `<Name>FuncTest`        | `<name>()` free function — correct values + autograd wiring      |
| `<Name>AutogradTest`    | End-to-end: `backward()` propagates correct gradients            |

This separation lets each tier fail independently. A `ForwardTest` failure means the math is wrong; a `FuncTest` failure means autograd wiring is broken even if the math is right.

---

## Best Practices

### 1. Use `operator()` for tensor access — never raw storage

```cpp
// Good
EXPECT_FLOAT_EQ(out(0), 2.0f);
EXPECT_FLOAT_EQ(out(i, j), ref);
EXPECT_FLOAT_EQ(x.grad()(i), 1.0f);

// Bad — bypasses the tensor's index logic and couples tests to internals
EXPECT_FLOAT_EQ(out.storage->data[0], 2.0f);
```

**Exception:** the `make_tensor` helper initialises a freshly-allocated contiguous tensor from a flat list where there is no higher-level API. `storage->data` is acceptable there and nowhere else.

### 2. `make_tensor` helper

Every test file that constructs tensors from literal values should define a file-local `make_tensor`:

```cpp
static Tensor make_tensor(std::initializer_list<size_t> dims,
                           std::initializer_list<float> vals,
                           bool requires_grad = false) {
    Shape s{};
    size_t i = 0;
    for (size_t d : dims) s[i++] = d;
    Tensor t(s, dims.size(), requires_grad);
    i = 0;
    for (float v : vals) t.storage->data[i++] = v;   // contiguous init
    return t;
}
```

For 2D tensors use explicit row/col loops with `t(i, j) = values[idx++]` instead of the flat storage write, so the access pattern is visible and self-documenting.

### 3. Cover non-contiguous inputs

Stride-aware code paths (anything with a `is_contiguous()` branch) must be tested with a transposed input. Use `tensor.T()` to produce a non-contiguous view:

```cpp
TEST_F(MulOpForwardTest, NonContiguousInputsViaTranspose) {
    auto A = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
    auto B = make_tensor({3, 2}, {1, 0, 0, 1, 0, 0});
    auto out = MulOp::forward(A, B.T());  // B.T() is non-contiguous
    // verify values...
}
```

Without this, the fast contiguous path can be correct while the general stride path silently computes wrong results.

### 4. Test non-unit upstream gradients

Backward tests with an all-ones upstream gradient (`Tensor::ones(...)`) cannot catch a missing `* grad[i]` factor. Always include at least one test with a non-uniform upstream gradient:

```cpp
auto grad = make_tensor({3}, {2.f, 3.f, 4.f});   // not all-ones
auto gx   = ReLUOp::backward(grad, x);
```

### 5. Test boundary values

For ops with conditional logic (relu threshold, tiling loop boundaries, broadcast dimensions):

- ReLU: test input exactly at `0.0f`
- Matmul: test shapes where `M`, `N`, or `K` is not a multiple of the tile size `T=64` (e.g., 65×65)
- Broadcasting: test the case where both dimensions are `1` simultaneously
- Extreme values: `sigmoid(-100)` should not produce NaN or `inf`

### 6. Test asserts fire on invalid input

Use `EXPECT_DEATH` for precondition violations:

```cpp
EXPECT_DEATH(MatMulOp::forward(a, b), "");  // shape mismatch
```

### 7. Do not test private implementation details

Tests call `Op::forward`, `Op::backward`, and the free functions. They do not access `autograd_meta` internals, `strides`, or `storage` except through the public API.

### 8. Keep `ForwardTest` independent of autograd

`ForwardTest` cases must never call `backward()` or check `autograd_meta`. The sole concern is correctness of the computed values. Autograd is `AutogradTest`'s domain.

### 9. Prefer `EXPECT_NEAR` over `EXPECT_FLOAT_EQ` for computed results

Transcendental functions (`exp`, `log`) accumulate floating-point error. Use `EXPECT_NEAR` with an appropriate tolerance:

```cpp
EXPECT_NEAR(out(i), ref, 1e-5f);
```

Reserve `EXPECT_FLOAT_EQ` for exact results (integer-valued arithmetic, identity operations).

---

## Common Pitfalls

| Pitfall | Why it matters |
|---------|---------------|
| Only testing contiguous inputs | Stride-aware slow paths never execute — bugs hide there |
| All-ones upstream gradient | Cannot detect a missing `* grad[i]` scale factor |
| Only testing 1D tensors in backward | 2D shape bookkeeping bugs go undetected |
| Checking `out.autograd_meta != nullptr` in `ForwardTest` | Forward must never attach autograd; test belongs in `FuncTest` |
| Using `storage->data` in assertions | Couples tests to layout; breaks if offset or strides change |
