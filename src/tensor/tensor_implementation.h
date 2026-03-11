#pragma once
#include <vector>
#include <memory>
#include <cassert>
#include <type_traits>
#include "storage.h"

inline std::vector<int64_t> default_strides(const std::vector<int64_t>& shape) {
    int n = shape.size();
    std::vector<int64_t> st(n, 1);
    for (int d = n - 2; d >= 0; --d)
        st[d] = st[d + 1] * shape[d + 1];
    return st;
}

struct TensorImpl {
    std::shared_ptr<Storage> storage;
    std::vector<int64_t>     shape;
    std::vector<int64_t>     strides;
    size_t                   offset = 0;

    TensorImpl(std::shared_ptr<Storage> s,
               std::vector<int64_t>     sz,
               std::vector<int64_t>     st  = {},
               size_t                   off = 0)
        : storage(std::move(s))
        , shape(std::move(sz))
        , strides(std::move(st))
        , offset(off)
    {
        if (strides.empty())
            strides = default_strides(shape);
    }

    // ----------------------------------------------------------------
    // Raw pointer access — use this in every op loop.
    //
    // Caller computes flat index directly:
    //   float* p = t.data_ptr();
    //   p[i * t.strides[0] + j * t.strides[1]] = val;
    //
    // Only valid for contiguous tensors (assert with is_contiguous()
    // in debug builds if you want the safety check).
    // ----------------------------------------------------------------
    float*       data_ptr()       { return storage->ptr() + offset; }
    const float* data_ptr() const { return storage->ptr() + offset; }

    // Convenience: raw stride values without going through the vector
    // abstraction each time — useful when caching strides in op loops.
    int64_t stride(int dim) const { return strides[dim]; }
    int64_t size  (int dim) const { return shape[dim];   }

    // ----------------------------------------------------------------
    // Variadic at<Idx...>(i, j, ...) — readable element access without
    // constructing a std::vector.
    //
    // The fold expression runs entirely at the call site; with -O2 it
    // compiles down to a handful of multiply-add instructions.
    //
    // Usage (same as before, just drop the braces):
    //   float v  = t.at(i, j);     // read
    //   t.at(i, j) = v;            // write
    // ----------------------------------------------------------------
    template<typename... Idx>
    float& at(Idx... indices) {
        static_assert((std::is_integral_v<Idx> && ...),
                      "at(): all indices must be integer types");
        assert(sizeof...(indices) == strides.size() &&
               "at(): wrong number of indices for tensor rank");
        int64_t flat = offset;
        int64_t dim  = 0;
        ((flat += static_cast<int64_t>(indices) * strides[dim++]), ...);
        return storage->ptr()[flat];
    }

    template<typename... Idx>
    float at(Idx... indices) const {
        static_assert((std::is_integral_v<Idx> && ...),
                      "at(): all indices must be integer types");
        assert(sizeof...(indices) == strides.size() &&
               "at(): wrong number of indices for tensor rank");
        int64_t flat = offset;
        int64_t dim  = 0;
        ((flat += static_cast<int64_t>(indices) * strides[dim++]), ...);
        return storage->ptr()[flat];
    }

    // ----------------------------------------------------------------
    // Metadata helpers
    // ----------------------------------------------------------------
    int64_t numel() const {
        if (shape.empty()) return 0;
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    int ndim() const { return static_cast<int>(shape.size()); }

    bool is_contiguous() const {
        return strides == default_strides(shape);
    }
};