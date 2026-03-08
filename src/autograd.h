#pragma once
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include "tensorlib.h"

struct Node;

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