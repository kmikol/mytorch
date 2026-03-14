#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "autograd.h"
#include "dataset/dataloader.h"
#include "dataset/mnist_dataset.h"
#include "layers/linear.h"
#include "loss_functions/cross_entropy.h"
#include "ops/activations/relu.h"
#include "optim/sgd.h"
#include "utils/metrics.h"

namespace {

std::string resolve_data_path(const std::string& rel_path) {
    const std::vector<std::string> candidates = {
        rel_path,
        "../" + rel_path,
        "/workspace/" + rel_path,
    };

    for (const auto& p : candidates) {
        if (std::filesystem::exists(p))
            return p;
    }
    return rel_path;
}

size_t argmax_row(const Tensor& x, size_t row) {
    const size_t cols = x.shape_at(1);
    size_t best_idx = 0;
    float best_val = x(row, 0);
    for (size_t j = 1; j < cols; ++j) {
        const float v = x(row, j);
        if (v > best_val) {
            best_val = v;
            best_idx = j;
        }
    }
    return best_idx;
}

}  // namespace

int main(int argc, char** argv) {
    const int epochs = (argc > 1) ? std::max(1, std::atoi(argv[1])) : 5;
    const size_t batch_size = 64;
    const float learning_rate = 0.1f;

    const std::string image_path = resolve_data_path("data/MNIST/train-images-idx3-ubyte");
    const std::string label_path = resolve_data_path("data/MNIST/train-labels-idx1-ubyte");

    if (!std::filesystem::exists(image_path) || !std::filesystem::exists(label_path)) {
        std::cerr << "MNIST files not found at expected path.\n";
        return 1;
    }

    MNISTDataset dataset(image_path, label_path);
    DataLoader loader(dataset, batch_size, /*shuffle=*/true, /*seed=*/2026u);

    if (!loader.has_next()) {
        std::cerr << "Dataset is empty.\n";
        return 1;
    }

    // Infer dimensions from a real batch rather than hardcoding image size.
    auto [probe_inputs, probe_targets] = loader.next_batch();
    const size_t input_features = probe_inputs.shape_at(1);
    const size_t num_classes = probe_targets.shape_at(1);

    Linear l1(input_features, 128);
    Linear l2(128, 64);
    Linear l3(64, num_classes);

    std::vector<Tensor*> params = l1.parameters();
    for (Tensor* p : l2.parameters()) params.push_back(p);
    for (Tensor* p : l3.parameters()) params.push_back(p);
    SGD optim(params, learning_rate);

    std::cout << "Training on full MNIST"
              << " | samples=" << dataset.size()
              << " | input_features=" << input_features
              << " | classes=" << num_classes
              << " | epochs=" << epochs
              << "\n";

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        loader.reset();

        float epoch_loss_sum = 0.0f;
        size_t batch_count = 0;
        std::vector<size_t> all_pred;
        std::vector<size_t> all_gt;
        all_pred.reserve(dataset.size());
        all_gt.reserve(dataset.size());

        while (loader.has_next()) {
            auto [inputs, targets] = loader.next_batch();

            Tensor h1     = relu(l1.forward(inputs));
            Tensor h2     = relu(l2.forward(h1));
            Tensor logits = l3.forward(h2);

            Tensor loss = cross_entropy(logits, targets);
            epoch_loss_sum += loss(0);
            ++batch_count;

            for (size_t b = 0; b < logits.shape_at(0); ++b) {
                all_pred.push_back(argmax_row(logits, b));
                all_gt.push_back(argmax_row(targets, b));
            }

            backward(loss);
            optim.step();
            optim.zero_grad();
        }

        const Metrics m = compute_metrics(all_pred, all_gt);
        const float mean_loss = epoch_loss_sum / static_cast<float>(batch_count);

        std::cout << "epoch=" << epoch
                  << " loss=" << mean_loss
                  << " accuracy=" << m.accuracy
                  << "\n";
    }

    return 0;
}