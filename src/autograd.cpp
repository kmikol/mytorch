#include "autograd.h"

#include <cassert>
#include <unordered_set>


std::shared_ptr<AutogradMeta> make_grad_meta(
    std::string name,
    std::vector<std::shared_ptr<AutogradMeta>> input_metas,
    std::function<std::vector<Tensor>(const Tensor&)> backward_fn)
{
    std::shared_ptr<Node> node = std::make_shared<Node>();
    node->name        = std::move(name);
    node->input_metas = std::move(input_metas);
    node->backward_fn = std::move(backward_fn);

    std::shared_ptr<AutogradMeta> meta = std::make_shared<AutogradMeta>();
    meta->requires_grad = true;
    meta->grad_fn       = node;
    return meta;
}


void backward(Tensor& output) {
    assert(output.requires_grad());

    // STEP 1: build execution order via post-order DFS (leaves first, output last)
    std::vector<AutogradMeta*> order;
    std::unordered_set<AutogradMeta*> visited;

    std::function<void(AutogradMeta*)> dfs = [&](AutogradMeta* meta) {
        if (meta == nullptr || visited.count(meta)) return;
        visited.insert(meta);
        if (meta->grad_fn)
            for (auto& input_meta : meta->grad_fn->input_metas)
                dfs(input_meta.get());
        order.push_back(meta);
    };

    dfs(output.autograd_meta.get());

    // STEP 2: seed output gradient with ones
    output.autograd_meta->grad = std::make_shared<Tensor>(
        Tensor::ones(output.shape, output.ndim)
    );

    // STEP 3: reverse sweep — propagate gradients from output back to leaves
    for (int i = static_cast<int>(order.size()) - 1; i >= 0; --i) {
        AutogradMeta* meta = order[i];

        if (!meta->grad || !meta->grad_fn) continue;

        // Disable graph construction during backward — we never need gradients
        // of gradients at this stage.
        NoGradGuard no_grad;

        std::vector<Tensor> input_grads = meta->grad_fn->backward_fn(*meta->grad);

        for (size_t j = 0; j < meta->grad_fn->input_metas.size(); ++j) {
            auto& input_meta = meta->grad_fn->input_metas[j];
            if (!input_meta || !input_meta->requires_grad) continue;

            if (!input_meta->grad) {
                // First gradient arriving at this node — store a contiguous copy.
                input_meta->grad = std::make_shared<Tensor>(input_grads[j].clone());
            } else {
                // Accumulate: grad tensors are always contiguous (created via clone/zeros).
                size_t n    = input_meta->grad->numel;
                float* dst  = input_meta->grad->storage->data;
                const float* src = input_grads[j].storage->data;
                for (size_t k = 0; k < n; ++k)
                    dst[k] += src[k];
            }
        }
    }
}
