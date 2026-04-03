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
#include "loss_functions/cross_entropy.h"
#include "networks/cnn.h"
#include "ops/activations/relu.h"
#include "ops/reshape.h"
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

Shape make_shape_4d(size_t d0, size_t d1, size_t d2, size_t d3) {
    Shape s{};
    s[0] = d0;
    s[1] = d1;
    s[2] = d2;
    s[3] = d3;
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    const int epochs = (argc > 1) ? std::max(1, std::atoi(argv[1])) : 5;
    const size_t batch_size = 64;
    const float learning_rate = 0.03f;

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
    const size_t image_rows = dataset.image_rows();
    const size_t image_cols = dataset.image_cols();

    if (input_features != image_rows * image_cols) {
        std::cerr << "Unexpected MNIST shape mismatch.\n";
        return 1;
    }

    CNN model(
        /*input_channels=*/1,
        image_rows,
        image_cols,
        /*conv_out_channels=*/8,
        /*kernel_h=*/3,
        /*kernel_w=*/3,
        relu,
        num_classes,
        /*stride_h=*/1,
        /*stride_w=*/1,
        /*padding_h=*/1,
        /*padding_w=*/1
    );

    std::vector<Tensor*> params = model.parameters();
    SGD optim(params, learning_rate);

    std::cout << "Training on full MNIST"
              << " | samples=" << dataset.size()
              << " | input_features=" << input_features
              << " | classes=" << num_classes
              << " | model=cnn"
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

            const size_t batch = inputs.shape_at(0);
            Tensor image_batch = reshape(inputs, make_shape_4d(batch, 1, image_rows, image_cols), 4);
            Tensor logits = model.forward(image_batch);

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