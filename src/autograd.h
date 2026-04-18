#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensor/tensor.h"

// Global flag — set to false during backward to avoid building a graph-within-a-graph.
inline bool grad_mode_enabled = true;

struct NoGradGuard {
    bool previous;
    NoGradGuard()  { previous = grad_mode_enabled; grad_mode_enabled = false; }
    ~NoGradGuard() { grad_mode_enabled = previous; }
    NoGradGuard(const NoGradGuard&)            = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
};

// Forward-declare Node so AutogradMeta can hold a shared_ptr<Node>.
struct Node;

// Attached to every tensor that participates in the autograd graph.
struct AutogradMeta {
    std::shared_ptr<Node>   grad_fn;        // null for leaf tensors
    std::shared_ptr<Tensor> grad;           // accumulated gradient
    bool                    requires_grad = false;
};

// A node in the computation graph. Each op creates one Node and stores
// pointers back to the AutogradMeta of its inputs.
struct Node {
    std::string name;
    std::vector<std::shared_ptr<AutogradMeta>> input_metas;
    std::function<std::vector<Tensor>(const Tensor&)> backward_fn;
};

// Helper used by ops to build a grad-function node and attach it to an output tensor.
// Declared here; implemented in autograd.cpp (to be added).
std::shared_ptr<AutogradMeta> make_grad_meta(
    std::string name,
    std::vector<std::shared_ptr<AutogradMeta>> input_metas,
    std::function<std::vector<Tensor>(const Tensor&)> backward_fn);

// Runs the reverse-mode sweep from `output` back to all leaf tensors.
// Declared here; implemented in autograd.cpp (to be added).
void backward(Tensor& output);
