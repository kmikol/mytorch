#pragma once
#include <vector>
#include <memory>
#include <cassert>
#include "tensor_implementation.h"
#include "utils.h"

// forward declaration — full definition is in autograd.h
struct AutogradMeta;

struct Tensor {
    std::shared_ptr<TensorImpl>   implementation;
    std::shared_ptr<AutogradMeta> autograd_meta;

    // ---- factories ----
    static Tensor fill(std::vector<int64_t> shape, float value,
                       bool requires_grad = false);

    static Tensor from_data(std::vector<float> data,
                            std::vector<int64_t> shape,
                            std::vector<int64_t> strides = {},
                            bool requires_grad = false);

    static Tensor zeros(std::vector<int64_t> shape, bool requires_grad = false);
    static Tensor ones (std::vector<int64_t> shape, bool requires_grad = false);

    // ---- shape ----
    int64_t numel()        const;
    int     ndim()         const;
    int64_t shape(int dim) const;

    // ---- element access ----
    float& at(const std::vector<int64_t>& idx);
    float  at(const std::vector<int64_t>& idx) const;

    // ---- autograd ----
    bool   requires_grad() const;
    bool   has_grad()      const;
    Tensor grad()          const;

    bool is_contiguous() const;

    // ---- clone ----
    Tensor clone() const;

    // ---- ops ----
    Tensor transpose(int dim0=-2, int dim1=-1) const;

    Tensor view(std::vector<int64_t> new_shape) const;

    Tensor contiguous() const;
};