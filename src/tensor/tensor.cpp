#include "tensor/tensor.h"
#include "autograd.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>


// ----------------------------------
//             Constructors
// ----------------------------------

// Private constructor: the single real constructor. All other constructors and
// factories route through here. Takes ownership of an already-allocated storage.
Tensor::Tensor(std::shared_ptr<Storage> storage, const Shape& shape, size_t ndim,
               bool requires_grad) {
    numel = 1;
    for (size_t i = 0; i < ndim; ++i) numel *= shape[i];

    strides       = strides_from_shape(shape, ndim);
    this->shape   = shape;
    this->ndim    = ndim;
    this->storage = std::move(storage);

    if (requires_grad) {
        autograd_meta = std::make_shared<AutogradMeta>();
        autograd_meta->requires_grad = true;
    }
}

// Validates ndim and computes numel — called before the delegating constructor
// argument expressions so the assert fires before any array access.
static std::shared_ptr<Storage> make_storage(const Shape& shape, size_t ndim) {
    assert(ndim <= MAX_DIM);
    size_t n = 1;
    for (size_t i = 0; i < ndim; ++i) n *= shape[i];
    return std::make_shared<Storage>(n);
}

// Public constructor: validates, allocates fresh storage, then delegates.
Tensor::Tensor(const Shape& shape, size_t ndim, bool requires_grad)
    : Tensor(make_storage(shape, ndim), shape, ndim, requires_grad) {}

Tensor Tensor::zeros(const Shape& shape, size_t ndim, bool requires_grad) {
    Tensor t(shape, ndim, requires_grad);
    t.storage->fill(0.f);
    return t;
}

Tensor Tensor::ones(const Shape& shape, size_t ndim, bool requires_grad) {
    Tensor t(shape, ndim, requires_grad);
    t.storage->fill(1.f);
    return t;
}

Tensor Tensor::from_storage(std::shared_ptr<Storage> storage,
                            const Shape& shape, size_t ndim) {
    assert(ndim <= MAX_DIM);
    size_t expected_numel = 1;
    for (size_t i = 0; i < ndim; ++i) expected_numel *= shape[i];
    assert(storage->size >= expected_numel);

    return Tensor(std::move(storage), shape, ndim);
}


// ----------------------------------
//             Checks
// ----------------------------------

bool Tensor::is_contiguous() const {
    return strides == strides_from_shape(shape, ndim);
}

Tensor Tensor::T() const {
    assert(ndim == 2);
    Tensor out          = *this;    // shallow copy: shared storage, all metadata copied
    out.autograd_meta   = nullptr;  // view is not a graph leaf
    std::swap(out.shape[0],   out.shape[1]);
    std::swap(out.strides[0], out.strides[1]);
    return out;
}


// ----------------------------------
//             Autograd
// ----------------------------------

bool Tensor::requires_grad() const {
    return autograd_meta != nullptr && autograd_meta->requires_grad;
}

bool Tensor::has_grad() const {
    return autograd_meta != nullptr && autograd_meta->grad != nullptr;
}

Tensor Tensor::grad() const {
    if (!has_grad())
        throw std::runtime_error("grad(): tensor has no gradient. "
                                 "Did you call backward() and set requires_grad=true?");
    return *autograd_meta->grad;
}


// ----------------------------------
//          Flat indexing
// ----------------------------------

// Maps logical flat index i to the correct storage position.
// For a contiguous tensor this is offset + i.
// For a non-contiguous tensor (e.g. transposed) we decompose i into
// per-dim indices using the contiguous strides, then apply the actual strides.
float& Tensor::flat(size_t i) {
    assert(i < numel);
    if (is_contiguous()) return storage->data[offset + i];
    size_t storage_idx = offset;
    size_t rem = i;
    Strides cs = strides_from_shape(shape, ndim);
    for (size_t d = 0; d < ndim; ++d) {
        size_t dim_idx  = rem / cs[d];
        rem            %= cs[d];
        storage_idx    += dim_idx * strides[d];
    }
    return storage->data[storage_idx];
}

const float& Tensor::flat(size_t i) const {
    assert(i < numel);
    if (is_contiguous()) return storage->data[offset + i];
    size_t storage_idx = offset;
    size_t rem = i;
    Strides cs = strides_from_shape(shape, ndim);
    for (size_t d = 0; d < ndim; ++d) {
        size_t dim_idx  = rem / cs[d];
        rem            %= cs[d];
        storage_idx    += dim_idx * strides[d];
    }
    return storage->data[storage_idx];
}


// ----------------------------------
//             Clone
// ----------------------------------

// Produces a contiguous deep copy with fresh storage and no autograd_meta.
// Handles non-contiguous sources (e.g. transposed tensors) by iterating logically.
Tensor Tensor::clone() const {
    Tensor out(shape, ndim);  // fresh contiguous storage, offset=0

    if (is_contiguous() && offset == 0) {
        std::copy(storage->data, storage->data + numel, out.storage->data);
    } else {
        // Decompose each output flat index into per-dim indices, then map
        // those indices through the source strides to find the source element.
        for (size_t flat = 0; flat < numel; ++flat) {
            size_t src_idx = offset;
            size_t rem     = flat;
            for (size_t d = 0; d < ndim; ++d) {
                size_t idx_d = rem / out.strides[d];
                rem         %= out.strides[d];
                src_idx     += idx_d * strides[d];
            }
            out.storage->data[flat] = storage->data[src_idx];
        }
    }
    return out;
}


// ----------------------------------
//            Utilities
// ----------------------------------

Strides Tensor::strides_from_shape(const Shape& shape, size_t ndim) {
    Strides strides{};
    strides[ndim - 1] = 1;
    for (size_t i = ndim - 1; i-- > 0;) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

Shape Tensor::shape_from_strides(const Strides& strides, size_t ndim, size_t numel) {
    Shape shape{};
    for (size_t i = 0; i < ndim - 1; ++i) {
        shape[i] = strides[i] / strides[i + 1];
    }
    size_t rest = 1;
    for (size_t i = 0; i < ndim - 1; ++i) rest *= shape[i];
    shape[ndim - 1] = numel / rest;
    return shape;
}

void Tensor::print() const {
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < ndim; ++i) {
        std::cout << shape[i];
        if (i < ndim - 1) std::cout << ", ";
    }
    std::cout << "], strides=[";
    for (size_t i = 0; i < ndim; ++i) {
        std::cout << strides[i];
        if (i < ndim - 1) std::cout << ", ";
    }
    std::cout << "], offset=" << offset << ")\n";

    if (ndim <= 2) {
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < (ndim > 1 ? shape[1] : 1); ++j) {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << "\n";
        }
    } else {
        std::cout << "Tensor data printing not implemented for ndim > 2\n";
    }
}
