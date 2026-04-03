#include <gtest/gtest.h>

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

static Shape make_shape_4d(size_t d0, size_t d1, size_t d2, size_t d3) {
    Shape s{};
    s[0] = d0;
    s[1] = d1;
    s[2] = d2;
    s[3] = d3;
    return s;
}

static Tensor to_nchw(const Tensor& flat_inputs, size_t rows, size_t cols) {
    const size_t B = flat_inputs.shape[0];
    Tensor out = reshape(flat_inputs, make_shape_4d(B, 1, rows, cols), 4);
    return out;
}

static float train_one_epoch(DataLoader& loader,
                             CNN& model,
                             SGD& optim,
                             size_t rows,
                             size_t cols) {
    loader.reset();

    float loss_sum = 0.0f;
    size_t batches = 0;

    while (loader.has_next()) {
        auto [inputs, targets] = loader.next_batch();

        Tensor image_batch = to_nchw(inputs, rows, cols);
        Tensor logits = model.forward(image_batch);

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

TEST(SmokeCNN, MNISTSubsample1000LossDecreasesOverFiveEpochs) {
    const std::string image_path = resolve_data_path("data/MNIST_subsamp1000/train-images-idx3-ubyte");
    const std::string label_path = resolve_data_path("data/MNIST_subsamp1000/train-labels-idx1-ubyte");

    ASSERT_TRUE(std::filesystem::exists(image_path));
    ASSERT_TRUE(std::filesystem::exists(label_path));

    MNISTDataset dataset(image_path, label_path);
    ASSERT_GT(dataset.size(), 0u);

    constexpr size_t batch_size = 32;
    DataLoader loader(dataset, batch_size, /*shuffle=*/true, /*seed=*/2026u);

    ASSERT_TRUE(loader.has_next());
    auto [probe_inputs, probe_targets] = loader.next_batch();
    ASSERT_EQ(probe_inputs.ndim, 2u);
    ASSERT_EQ(probe_targets.ndim, 2u);

    const size_t input_features = probe_inputs.shape_at(1);
    const size_t num_classes = probe_targets.shape_at(1);
    const size_t rows = dataset.image_rows();
    const size_t cols = dataset.image_cols();
    ASSERT_EQ(input_features, rows * cols);

    CNN model(
        /*input_channels=*/1,
        rows,
        cols,
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

    SGD optim(model.parameters(), /*learning_rate=*/0.03f);

    float first_epoch_loss = 0.0f;
    float last_epoch_loss = 0.0f;

    for (int epoch = 0; epoch < 5; ++epoch) {
        const float epoch_loss = train_one_epoch(loader, model, optim, rows, cols);
        std::cout << "cnn epoch=" << epoch + 1 << " loss=" << epoch_loss << "\n";

        if (epoch == 0)
            first_epoch_loss = epoch_loss;
        if (epoch == 4)
            last_epoch_loss = epoch_loss;
    }

    EXPECT_LT(last_epoch_loss, first_epoch_loss);
}
