#pragma once
#include <vector>
#include <memory>
#include <cassert>
#include <type_traits>
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
    static Tensor from_data(std::vector<float>   data,
                            std::vector<int64_t> shape,
                            std::vector<int64_t> strides = {},
                            bool requires_grad = false);
    static Tensor zeros(std::vector<int64_t> shape, bool requires_grad = false);
    static Tensor ones (std::vector<int64_t> shape, bool requires_grad = false);

    // ---- shape ----
    int64_t numel()        const;
    int     ndim()         const;
    int64_t shape(int dim) const;
    int64_t stride(int dim) const;

    std::vector<int64_t> shape () const {
        return implementation->shape;
    }
    std::vector<int64_t> strides() const {
        return implementation->strides;
    }

    // ---- raw data access — use in op loops ----
    // Only valid for contiguous tensors. Caller is responsible for
    // computing flat indices via stride(d).
    float*       data_ptr()       { return implementation->data_ptr(); }
    const float* data_ptr() const { return implementation->data_ptr(); }

    // ---- element access — readable single-element access outside hot loops ----
    // Replaces at(const std::vector<int64_t>&). No heap allocation.
    // Usage: t.at(i, j)  instead of the old t.at(i, j)
    template<typename... Idx>
    float& at(Idx... indices) {
        static_assert((std::is_integral_v<Idx> && ...),
                      "at(): all indices must be integer types");
        return implementation->at(indices...);
    }

    template<typename... Idx>
    float at(Idx... indices) const {
        static_assert((std::is_integral_v<Idx> && ...),
                      "at(): all indices must be integer types");
        return implementation->at(indices...);
    }

    // ---- autograd ----
    bool   requires_grad() const;
    bool   has_grad()      const;
    Tensor grad()          const;

    // ---- layout ----
    bool is_contiguous() const;

    // ---- clone ----
    Tensor clone() const;

    // ---- ops ----
    Tensor transpose(int dim0 = -2, int dim1 = -1) const;
    Tensor view(std::vector<int64_t> new_shape)     const;
    Tensor reshape(std::vector<int64_t> new_shape)  const;
    Tensor contiguous()                              const;
};