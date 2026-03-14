#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
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

namespace {

static std::string resolve_data_path(const std::string& rel_path) {
    const std::vector<std::string> candidates = {
        rel_path,
        "../" + rel_path,
        "/workspace/" + rel_path,
    };

    for (const auto& path : candidates) {
        if (std::filesystem::exists(path))
            return path;
    }

    return rel_path;
}

static float train_one_epoch(DataLoader& loader,
                             Linear& l1,
                             Linear& l2,
                             SGD& optim) {
    loader.reset();

    float loss_sum = 0.0f;
    size_t batches = 0;

    while (loader.has_next()) {
        auto [inputs, targets] = loader.next_batch();

        Tensor h = relu(l1.forward(inputs));
        Tensor logits = l2.forward(h);

        Tensor loss = cross_entropy(logits, targets);
        loss_sum += loss(0);
        ++batches;

        backward(loss);
        optim.step();
        optim.zero_grad();
    }

    return loss_sum / static_cast<float>(batches);
}

}  // namespace

TEST(SmokeMLP, MNISTSubsample1000LossDecreasesOverTenEpochs) {
    const std::string image_path = resolve_data_path("data/MNIST_subsamp1000/train-images-idx3-ubyte");
    const std::string label_path = resolve_data_path("data/MNIST_subsamp1000/train-labels-idx1-ubyte");

    ASSERT_TRUE(std::filesystem::exists(image_path));
    ASSERT_TRUE(std::filesystem::exists(label_path));

    MNISTDataset dataset(image_path, label_path);
    ASSERT_GT(dataset.size(), 0u);

    constexpr size_t batch_size = 32;
    DataLoader loader(dataset, batch_size, /*shuffle=*/true, /*seed=*/2026u);

    // Infer input size from an actual batch produced by the loader.
    ASSERT_TRUE(loader.has_next());
    auto [probe_inputs, probe_targets] = loader.next_batch();
    ASSERT_EQ(probe_inputs.ndim, 2u);
    ASSERT_EQ(probe_targets.ndim, 2u);

    const size_t input_features = probe_inputs.shape_at(1);
    const size_t num_classes = probe_targets.shape_at(1);

    Linear l1(input_features, 64);
    Linear l2(64, num_classes);

    std::vector<Tensor*> params = l1.parameters();
    const auto l2_params = l2.parameters();
    params.insert(params.end(), l2_params.begin(), l2_params.end());

    SGD optim(params, /*learning_rate=*/0.1f);

    float first_epoch_loss = 0.0f;
    float last_epoch_loss = 0.0f;

    for (int epoch = 0; epoch < 10; ++epoch) {
        const float epoch_loss = train_one_epoch(loader, l1, l2, optim);
        std::cout << "epoch=" << epoch + 1 << " loss=" << epoch_loss << "\n";

        if (epoch == 0)
            first_epoch_loss = epoch_loss;
        if (epoch == 9)
            last_epoch_loss = epoch_loss;
    }

    EXPECT_LT(last_epoch_loss, first_epoch_loss);
}
