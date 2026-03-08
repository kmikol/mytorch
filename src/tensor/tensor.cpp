#include "tensor.h"
#include "autograd.h"  
#include "ops/ops.h"   

// ---- factories ----

Tensor Tensor::fill(std::vector<int64_t> shape, float value, bool requires_grad) {
    Tensor tensor;

    int64_t n = 1;
    for (int64_t dim : shape) n *= dim;

    auto storage_ptr = std::make_shared<Storage>(n, value);
    tensor.implementation = std::make_shared<TensorImpl>(storage_ptr, std::move(shape));

    if (requires_grad) {
        tensor.autograd_meta = std::make_shared<AutogradMeta>();
        tensor.autograd_meta->requires_grad = true;
    }

    return tensor;
}

Tensor Tensor::from_data(std::vector<float> data,
                          std::vector<int64_t> shape,
                          std::vector<int64_t> strides,
                          bool requires_grad) {
    Tensor tensor;

    Storage storage;
    storage.data = std::move(data);
    auto storage_ptr = std::make_shared<Storage>(std::move(storage));

    tensor.implementation = std::make_shared<TensorImpl>(storage_ptr, shape, strides);

    if (requires_grad) {
        tensor.autograd_meta = std::make_shared<AutogradMeta>();
        tensor.autograd_meta->requires_grad = true;
    }

    return tensor;
}

Tensor Tensor::zeros(std::vector<int64_t> shape, bool requires_grad) {
    return fill(shape, 0.f, requires_grad);
}

Tensor Tensor::ones(std::vector<int64_t> shape, bool requires_grad) {
    return fill(shape, 1.f, requires_grad);
}

// ---- shape ----

int64_t Tensor::numel()        const { return implementation->numel(); }
int     Tensor::ndim()         const { return implementation->ndim(); }
int64_t Tensor::shape(int dim) const { return implementation->shape[dim]; }

// ---- element access ----

float& Tensor::at(const std::vector<int64_t>& idx)       { return implementation->at(idx); }
float  Tensor::at(const std::vector<int64_t>& idx) const { return implementation->at(idx); }

// ---- autograd ----
// these methods need AutogradMeta to be fully defined
// that's why they live in the .cpp and not inline in the header

bool Tensor::requires_grad() const {
    return autograd_meta != nullptr && autograd_meta->requires_grad;
}

bool Tensor::has_grad() const {
    return autograd_meta != nullptr && autograd_meta->grad != nullptr;
}

Tensor Tensor::grad() const {
    assert(has_grad());
    return *autograd_meta->grad;   // needs full AutogradMeta definition
}

bool Tensor::is_contiguous() const {
    return implementation->is_contiguous();
}

// ---- clone ----

Tensor Tensor::clone() const {
    
    if (this->is_contiguous()) {
        // fast path for contiguous tensors — can copy data directly
        std::vector<float> new_data = implementation->storage->data; // copy constructor
        return from_data(std::move(new_data), implementation->get_shape());
    } else {
        // general path for non-contiguous tensors — copy element by element
        return this->contiguous();
    }
}

// ---- ops ----
Tensor Tensor::transpose(int dim0, int dim1) const {
    return ::transpose(*this, dim0, dim1);
}

Tensor Tensor::view(std::vector<int64_t> new_shape) const {
    return ::view(*this, new_shape);
}

Tensor Tensor::contiguous() const {
    return ::contiguous(*this);
}