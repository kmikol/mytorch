#pragma once
#include <vector>
#include <memory>
#include <cassert>

#include "storage.h"

inline std::vector<int64_t> default_strides(const std::vector<int64_t>& shape) {
    int n = shape.size();
    std::vector<int64_t> st(n, 1);
    for (int d = n - 2; d >= 0; --d)
        st[d] = st[d+1] * shape[d+1];
    return st;
}

struct TensorImpl {
    std::shared_ptr<Storage> storage;
    std::vector<int64_t> shape;    // e.g. {3, 4} for a 3×4 tensor
    std::vector<int64_t> strides;  // e.g. {4, 1} for row-major
    size_t offset = 0;

    
    TensorImpl(std::shared_ptr<Storage> s,
        std::vector<int64_t> sz,
        std::vector<int64_t> st = {},
        size_t off = 0
    ) : storage(std::move(s)), shape(std::move(sz)), strides(std::move(st)), offset(off) {

        // compute default strides if not provided
        if (strides.empty()) {
            strides = default_strides(shape);
        }
    }


    float& at(const std::vector<int64_t>& idx) {
        int64_t flat_index = offset;
        for (size_t i = 0; i < idx.size(); ++i) {
            flat_index += idx[i] * strides[i];
        }
        return storage->ptr()[flat_index];
    }

    float at(const std::vector<int64_t>& idx) const {
        int64_t flat_index = offset;
        for (size_t i = 0; i < idx.size(); ++i) {
            flat_index += idx[i] * strides[i];
        }
        return storage->ptr()[flat_index];
    }


    int64_t numel() const {

        if (shape.empty()) return 0;

        int64_t n = 1;
        for (size_t i = 0; i < shape.size(); ++i) n *= shape[i];

        return n;
    }

    int ndim() const {
        return shape.size();
    }

    std::vector<int64_t> get_shape() {
        return shape;
    }

    std::vector<int64_t> get_strides() {
        return strides;
    }

    // a tensor is contiguous if its strides exactly match what default_strides
    // would produce for its current shape — meaning elements are sequential
    // in memory with no gaps or reordering
    //
    // this matters because raw pointer arithmetic only works correctly
    // on contiguous tensors — if strides don't match row-major order,
    // skipping through memory by shape*stride gives wrong elements
    bool is_contiguous() const {
        return strides == default_strides(shape);
    }
};