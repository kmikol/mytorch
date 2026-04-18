#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "tensor/storage.h"


#define MAX_DIM 8
using Shape   = std::array<size_t, MAX_DIM>;
using Strides = std::array<size_t, MAX_DIM>;

// Full definition lives in autograd.h, which depends on Tensor being already defined.
// Forward-declare here so Tensor can hold a shared_ptr<AutogradMeta> without a circular include.
struct AutogradMeta;


class Tensor {

    public:
        std::shared_ptr<Storage>      storage;
        std::shared_ptr<AutogradMeta> autograd_meta;  // null unless requires_grad
        Shape   shape;
        Strides strides;
        size_t  ndim;
        size_t  numel;
        size_t  offset = 0;


        // ----------------------------------
        //             Constructors
        // ----------------------------------

        Tensor(const Shape& shape, size_t ndim, bool requires_grad = false);

        // Factory: all-zeros tensor.
        static Tensor zeros(const Shape& shape, size_t ndim, bool requires_grad = false);

        // Factory: all-ones tensor.
        static Tensor ones(const Shape& shape, size_t ndim, bool requires_grad = false);

        // Factory: wrap an existing Storage object.
        // Useful for constructing tensors from pre-populated data without element-wise writes.
        // Asserts that storage->size >= numel implied by shape/ndim.
        static Tensor from_storage(std::shared_ptr<Storage> storage,
                                   const Shape& shape, size_t ndim);

        // ----------------------------------
        //             Operators
        // ----------------------------------

        // Multi-dimensional element access via operator()(i, j, k, ...).
        // Compile-time check that all indices are integers; runtime assert on rank match.
        template<typename... Idx>
        float& operator()(Idx... indices) {
            static_assert((std::is_integral_v<Idx> && ...), "indices must be integers");
            assert(sizeof...(indices) == ndim);

            size_t curr_dim = 0;
            size_t flat_idx = offset;
            ((flat_idx += static_cast<size_t>(indices) * strides[curr_dim++]), ...);

            return (*storage)[flat_idx];
        }

        template<typename... Idx>
        const float& operator()(Idx... indices) const {
            static_assert((std::is_integral_v<Idx> && ...), "indices must be integers");
            assert(sizeof...(indices) == ndim);

            size_t curr_dim = 0;
            size_t flat_idx = offset;
            ((flat_idx += static_cast<size_t>(indices) * strides[curr_dim++]), ...);

            return (*storage)[flat_idx];
        }

        // ----------------------------------
        //          Per-dim accessors
        // ----------------------------------

        // Bounds-checked access to shape[dim]. Throws std::out_of_range if dim >= ndim.
        size_t shape_at(size_t dim) const {
            if (dim >= ndim)
                throw std::out_of_range("shape_at: dim " + std::to_string(dim) +
                                        " out of range for ndim " + std::to_string(ndim));
            return shape[dim];
        }

        // Bounds-checked access to strides[dim]. Throws std::out_of_range if dim >= ndim.
        size_t stride_at(size_t dim) const {
            if (dim >= ndim)
                throw std::out_of_range("stride_at: dim " + std::to_string(dim) +
                                        " out of range for ndim " + std::to_string(ndim));
            return strides[dim];
        }

        // ----------------------------------
        //             Checks
        // ----------------------------------

        bool is_contiguous() const;

        // Returns a 2D transposed view: shape and strides for dims 0 and 1 are
        // swapped. No data is copied — this is O(1).
        // The returned tensor shares storage with *this; autograd_meta is not
        // propagated (the view is intended for use inside op kernels, not for
        // building a graph node).
        Tensor T() const;

        // ----------------------------------
        //             Autograd
        // ----------------------------------

        bool   requires_grad() const;
        bool   has_grad()      const;
        Tensor grad()          const;

        // ----------------------------------
        //             Clone
        // ----------------------------------

        // Deep copy: new storage, same logical values, contiguous layout, no autograd_meta.
        Tensor clone() const;

        // ----------------------------------
        //            Utilities
        // ----------------------------------

        static Strides strides_from_shape(const Shape& shape, size_t ndim);
        static Shape   shape_from_strides(const Strides& strides, size_t ndim, size_t numel);

        void print() const;

    private:
        // Single real constructor: wires up all metadata and takes ownership of storage.
        // The public constructor delegates here after allocating its own storage.
        // from_storage() calls this directly to wrap an existing buffer with no allocation.
        Tensor(std::shared_ptr<Storage> storage, const Shape& shape, size_t ndim,
               bool requires_grad = false);
};
