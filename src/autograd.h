#pragma once
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include "tensorlib.h"

// global flag controlling whether ops register gradient nodes
// true by default — ops build the graph normally
// set to false during backward to skip graph building entirely
inline bool grad_mode_enabled = true;

struct Node;

struct NoGradGuard {
    bool previous;
    NoGradGuard()  { previous = grad_mode_enabled; grad_mode_enabled = false; }
    ~NoGradGuard() { grad_mode_enabled = previous; }
    NoGradGuard(const NoGradGuard&)            = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
};

// forward declaration — full definition below, after Tensor is known
struct AutogradMeta {
    std::shared_ptr<Node>   grad_fn;
    std::shared_ptr<Tensor> grad;
    bool requires_grad = false;
};

struct Node {
    std::string name;
    std::vector<std::shared_ptr<AutogradMeta>> input_metas;
    std::function<std::vector<Tensor>(const Tensor&)> backward_fn;
};

// declarations only — bodies are in autograd.cpp
std::shared_ptr<AutogradMeta> make_grad_meta(
    std::string name,
    std::vector<std::shared_ptr<AutogradMeta>> input_metas,
    std::function<std::vector<Tensor>(const Tensor&)> backward_fn);

void backward(Tensor& output);

