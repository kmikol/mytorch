#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "dataset/dataloader.h"
#include "dataset/mnist_dataset.h"

namespace {

struct SplitConfig {
    std::string folder;
    std::string split;
    uint32_t rows;
    uint32_t cols;
};

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

size_t argmax_one_hot_row(const Tensor& t, size_t row) {
    const size_t classes = t.shape_at(1);
    size_t best = 0;
    float best_v = t(row, 0);
    for (size_t j = 1; j < classes; ++j) {
        if (t(row, j) > best_v) {
            best_v = t(row, j);
            best = j;
        }
    }
    return best;
}

void run_and_print_stats(const SplitConfig& cfg) {
    const std::string image_rel = cfg.folder + "/" + cfg.split + "-images-idx3-ubyte";
    const std::string label_rel = cfg.folder + "/" + cfg.split + "-labels-idx1-ubyte";

    const std::string image_path = resolve_data_path(image_rel);
    const std::string label_path = resolve_data_path(label_rel);

    ASSERT_TRUE(std::filesystem::exists(image_path));
    ASSERT_TRUE(std::filesystem::exists(label_path));

    MNISTDataset ds(image_path, label_path);
    ASSERT_GT(ds.size(), 0u);
    EXPECT_EQ(ds.image_rows(), cfg.rows);
    EXPECT_EQ(ds.image_cols(), cfg.cols);

    DataLoader loader(ds, /*batch_size=*/128, /*shuffle=*/false);
    ASSERT_TRUE(loader.has_next());

    auto [probe_x, probe_y] = loader.next_batch();
    ASSERT_EQ(probe_x.ndim, 2u);
    ASSERT_EQ(probe_y.ndim, 2u);

    const size_t input_features = probe_x.shape_at(1);
    const size_t target_features = probe_y.shape_at(1);
    const size_t expected_features = static_cast<size_t>(cfg.rows) * static_cast<size_t>(cfg.cols);

    EXPECT_EQ(input_features, expected_features);
    EXPECT_EQ(target_features, 10u);

    loader.reset();

    std::array<size_t, 10> class_counts{};
    size_t seen_samples = 0;

    while (loader.has_next()) {
        auto [inputs, targets] = loader.next_batch();
        ASSERT_EQ(inputs.shape_at(1), input_features);
        ASSERT_EQ(targets.shape_at(1), 10u);

        for (size_t b = 0; b < inputs.shape_at(0); ++b) {
            ++seen_samples;
            const size_t label = argmax_one_hot_row(targets, b);
            ASSERT_LT(label, class_counts.size());
            class_counts[label] += 1;
        }
    }

    EXPECT_EQ(seen_samples, ds.size());

    size_t sum_dist = 0;
    for (size_t c : class_counts)
        sum_dist += c;
    EXPECT_EQ(sum_dist, seen_samples);

    std::cout << "\n[dataset] " << cfg.folder << " split=" << cfg.split
              << " samples=" << seen_samples
              << " shape=" << cfg.rows << "x" << cfg.cols
              << " input_features=" << input_features
              << " classes=" << target_features
              << " class_distribution=";

    for (size_t k = 0; k < class_counts.size(); ++k) {
        std::cout << k << ":" << class_counts[k];
        if (k + 1 != class_counts.size())
            std::cout << ",";
    }
    std::cout << "\n";
}

}  // namespace

TEST(DatasetMNISTStats, FullMNISTTrain) {
    run_and_print_stats({"data/MNIST", "train", 28u, 28u});
}

TEST(DatasetMNISTStats, FullMNISTTest) {
    run_and_print_stats({"data/MNIST", "t10k", 28u, 28u});
}

TEST(DatasetMNISTStats, Subsample100Train) {
    run_and_print_stats({"data/MNIST_subsamp100", "train", 14u, 14u});
}

TEST(DatasetMNISTStats, Subsample100Test) {
    run_and_print_stats({"data/MNIST_subsamp100", "t10k", 14u, 14u});
}

TEST(DatasetMNISTStats, Subsample1000Train) {
    run_and_print_stats({"data/MNIST_subsamp1000", "train", 14u, 14u});
}

TEST(DatasetMNISTStats, Subsample1000Test) {
    run_and_print_stats({"data/MNIST_subsamp1000", "t10k", 14u, 14u});
}
