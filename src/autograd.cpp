#include "autograd.h"
#include <unordered_set>
#include <cassert>

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

    // STEP 1: topological sort via DFS
    std::vector<AutogradMeta*> order;
    std::unordered_set<AutogradMeta*> visited;

    std::function<void(AutogradMeta*)> dfs = [&](AutogradMeta* meta) {
        if (meta == nullptr) return;
        if (visited.count(meta) > 0) return;
        visited.insert(meta);
        if (meta->grad_fn != nullptr)
            for (auto& input_meta : meta->grad_fn->input_metas)
                dfs(input_meta.get());
        order.push_back(meta);
    };

    dfs(output.autograd_meta.get());

    // STEP 2: seed the output gradient with ones
    output.autograd_meta->grad = std::make_shared<Tensor>(
        Tensor::ones(output.shape())
    );

    // STEP 3: reverse sweep
    for (int i = (int)order.size() - 1; i >= 0; --i) {
        AutogradMeta* meta = order[i];

        if (meta->grad == nullptr)    continue;
        if (meta->grad_fn == nullptr) continue;

        std::vector<Tensor> input_grads = meta->grad_fn->backward_fn(*meta->grad);

        for (size_t j = 0; j < meta->grad_fn->input_metas.size(); ++j) {
            std::shared_ptr<AutogradMeta>& input_meta = meta->grad_fn->input_metas[j];

            if (input_meta == nullptr)            continue;
            if (!input_meta->requires_grad)       continue;

            if (input_meta->grad == nullptr) {
                input_meta->grad = std::make_shared<Tensor>(input_grads[j].clone());
            } else {
                for (int64_t r = 0; r < input_meta->grad->shape(0); ++r)
                    for (int64_t c = 0; c < input_meta->grad->shape(1); ++c)
                        input_meta->grad->at(r, c) += input_grads[j].at(r, c);
            }
        }
    }
}
