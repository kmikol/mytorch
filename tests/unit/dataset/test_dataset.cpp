#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "dataset/dataloader.h"
#include "dataset/mnist_dataset.h"

namespace {

static Shape make_shape_2d(size_t d0, size_t d1) {
    Shape s{};
    s[0] = d0;
    s[1] = d1;
    return s;
}

static std::string resolve_data_path(const std::string& rel_path) {
    const std::array<std::string, 4> candidates = {
        rel_path,
        "../" + rel_path,
        "/workspace/" + rel_path,
        std::string("/workspace/build/") + "../" + rel_path
    };

    for (const auto& path : candidates) {
        if (std::filesystem::exists(path))
            return path;
    }

    ADD_FAILURE() << "Unable to locate dataset file: " << rel_path;
    return rel_path;
}

static uint32_t read_be_uint32(std::ifstream& file) {
    unsigned char bytes[4]{};
    file.read(reinterpret_cast<char*>(bytes), 4);
    EXPECT_TRUE(file.good());
    return (static_cast<uint32_t>(bytes[0]) << 24)
         | (static_cast<uint32_t>(bytes[1]) << 16)
         | (static_cast<uint32_t>(bytes[2]) << 8)
         |  static_cast<uint32_t>(bytes[3]);
}

struct ImageHeader {
    uint32_t magic = 0;
    uint32_t count = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
};

struct LabelHeader {
    uint32_t magic = 0;
    uint32_t count = 0;
};

static ImageHeader read_image_header(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    EXPECT_TRUE(in.is_open()) << path;

    ImageHeader h{};
    h.magic = read_be_uint32(in);
    h.count = read_be_uint32(in);
    h.rows = read_be_uint32(in);
    h.cols = read_be_uint32(in);
    return h;
}

static LabelHeader read_label_header(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    EXPECT_TRUE(in.is_open()) << path;

    LabelHeader h{};
    h.magic = read_be_uint32(in);
    h.count = read_be_uint32(in);
    return h;
}

static uint8_t read_label_at(const std::string& label_path, size_t index) {
    std::ifstream in(label_path, std::ios::binary);
    EXPECT_TRUE(in.is_open()) << label_path;

    const std::streamoff header_size = 8;
    in.seekg(header_size + static_cast<std::streamoff>(index), std::ios::beg);
    EXPECT_TRUE(in.good());

    unsigned char label = 0;
    in.read(reinterpret_cast<char*>(&label), 1);
    EXPECT_TRUE(in.good());
    return label;
}

static std::vector<uint8_t> read_image_pixels_prefix(const std::string& image_path,
                                                     size_t index,
                                                     size_t pixels_per_image,
                                                     size_t prefix_len) {
    std::ifstream in(image_path, std::ios::binary);
    EXPECT_TRUE(in.is_open()) << image_path;

    const size_t n = std::min(prefix_len, pixels_per_image);
    const std::streamoff header_size = 16;
    const std::streamoff image_offset = static_cast<std::streamoff>(index * pixels_per_image);
    in.seekg(header_size + image_offset, std::ios::beg);
    EXPECT_TRUE(in.good());

    std::vector<uint8_t> pixels(n, 0);
    in.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(n));
    EXPECT_TRUE(in.good());
    return pixels;
}

class ToyDataset final : public Dataset {
public:
    ToyDataset(size_t sample_count, size_t input_features, size_t target_features) {
        samples.reserve(sample_count);
        for (size_t i = 0; i < sample_count; ++i) {
            Tensor input = Tensor::zeros(make_shape_2d(1, input_features), 2);
            Tensor target = Tensor::zeros(make_shape_2d(1, target_features), 2);

            // Encode the sample index in feature 0 to make ordering observable.
            input(0, 0) = static_cast<float>(i);
            for (size_t f = 1; f < input_features; ++f)
                input(0, f) = static_cast<float>(i * 10 + f);

            target(0, i % target_features) = 1.0f;
            samples.push_back({input, target});
        }
    }

    size_t size() const override {
        return samples.size();
    }

    Sample get(size_t index) const override {
        assert(index < samples.size());
        return samples[index];
    }

private:
    std::vector<Sample> samples;
};

static std::vector<size_t> collect_batch_order(DataLoader& loader) {
    std::vector<size_t> order;
    while (loader.has_next()) {
        auto [inputs, targets] = loader.next_batch();
        EXPECT_EQ(inputs.ndim, 2u);
        EXPECT_EQ(targets.ndim, 2u);

        for (size_t b = 0; b < inputs.shape_at(0); ++b)
            order.push_back(static_cast<size_t>(inputs(b, 0)));
    }
    return order;
}

static std::vector<size_t> shuffled_indices(size_t n, uint32_t seed, size_t rounds) {
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0u);

    std::mt19937 rng(seed);
    for (size_t r = 0; r < rounds; ++r)
        std::shuffle(idx.begin(), idx.end(), rng);

    return idx;
}

static void expect_one_hot_row(const Tensor& target_row) {
    ASSERT_EQ(target_row.ndim, 2u);
    ASSERT_EQ(target_row.shape_at(0), 1u);
    ASSERT_EQ(target_row.shape_at(1), 10u);

    size_t ones = 0;
    for (size_t k = 0; k < 10; ++k) {
        const float v = target_row(0, k);
        EXPECT_TRUE(v == 0.0f || v == 1.0f);
        if (v == 1.0f)
            ++ones;
    }
    EXPECT_EQ(ones, 1u);
}

static void verify_split_with_real_files(const std::string& folder,
                                         const std::string& split_prefix,
                                         uint32_t expected_rows,
                                         uint32_t expected_cols,
                                         size_t pixel_prefix_to_check) {
    const std::string image_rel = folder + "/" + split_prefix + "-images-idx3-ubyte";
    const std::string label_rel = folder + "/" + split_prefix + "-labels-idx1-ubyte";

    const std::string image_path = resolve_data_path(image_rel);
    const std::string label_path = resolve_data_path(label_rel);

    const ImageHeader ih = read_image_header(image_path);
    const LabelHeader lh = read_label_header(label_path);

    EXPECT_EQ(ih.magic, 2051u);
    EXPECT_EQ(lh.magic, 2049u);
    EXPECT_EQ(ih.rows, expected_rows);
    EXPECT_EQ(ih.cols, expected_cols);
    EXPECT_EQ(ih.count, lh.count);

    MNISTDataset ds(image_path, label_path);

    EXPECT_EQ(ds.size(), static_cast<size_t>(ih.count));
    EXPECT_EQ(ds.image_rows(), expected_rows);
    EXPECT_EQ(ds.image_cols(), expected_cols);
    EXPECT_EQ(ds.input_size(), static_cast<size_t>(expected_rows) * static_cast<size_t>(expected_cols));
    EXPECT_EQ(ds.num_classes(), 10u);

    ASSERT_GT(ds.size(), 2u);

    const std::vector<size_t> probe_indices = {0u, ds.size() / 2u, ds.size() - 1u};
    const size_t pixels_per_image = static_cast<size_t>(expected_rows) * static_cast<size_t>(expected_cols);

    for (size_t idx : probe_indices) {
        const Sample s = ds.get(idx);

        ASSERT_EQ(s.input.ndim, 2u);
        ASSERT_EQ(s.input.shape_at(0), 1u);
        ASSERT_EQ(s.input.shape_at(1), pixels_per_image);

        expect_one_hot_row(s.target);

        for (size_t p = 0; p < pixels_per_image; ++p) {
            const float v = s.input(0, p);
            EXPECT_GE(v, 0.0f);
            EXPECT_LE(v, 1.0f);
        }

        const uint8_t label = read_label_at(label_path, idx);
        ASSERT_LT(label, 10u);
        for (size_t k = 0; k < 10; ++k) {
            const float expected = (k == label) ? 1.0f : 0.0f;
            EXPECT_FLOAT_EQ(s.target(0, k), expected);
        }

        const auto raw_pixels = read_image_pixels_prefix(image_path, idx, pixels_per_image, pixel_prefix_to_check);
        for (size_t p = 0; p < raw_pixels.size(); ++p) {
            const float expected = static_cast<float>(raw_pixels[p]) / 255.0f;
            EXPECT_NEAR(s.input(0, p), expected, 1e-7f);
        }
    }
}

}  // namespace

class DataLoaderTest : public ::testing::Test {};

TEST_F(DataLoaderTest, NoShufflePreservesOrderAndShapes) {
    ToyDataset ds(/*sample_count=*/10, /*input_features=*/4, /*target_features=*/3);
    DataLoader loader(ds, /*batch_size=*/4, /*shuffle=*/false);

    ASSERT_TRUE(loader.has_next());

    auto [x1, y1] = loader.next_batch();
    EXPECT_EQ(x1.shape_at(0), 4u);
    EXPECT_EQ(x1.shape_at(1), 4u);
    EXPECT_EQ(y1.shape_at(0), 4u);
    EXPECT_EQ(y1.shape_at(1), 3u);
    EXPECT_FLOAT_EQ(x1(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(x1(3, 0), 3.0f);

    auto [x2, y2] = loader.next_batch();
    EXPECT_EQ(x2.shape_at(0), 4u);
    EXPECT_EQ(y2.shape_at(0), 4u);
    EXPECT_FLOAT_EQ(x2(0, 0), 4.0f);
    EXPECT_FLOAT_EQ(x2(3, 0), 7.0f);

    auto [x3, y3] = loader.next_batch();
    EXPECT_EQ(x3.shape_at(0), 2u);
    EXPECT_EQ(y3.shape_at(0), 2u);
    EXPECT_FLOAT_EQ(x3(0, 0), 8.0f);
    EXPECT_FLOAT_EQ(x3(1, 0), 9.0f);

    EXPECT_FALSE(loader.has_next());
}

TEST_F(DataLoaderTest, ResetWithoutShuffleRewindsIdentically) {
    ToyDataset ds(/*sample_count=*/6, /*input_features=*/2, /*target_features=*/2);
    DataLoader loader(ds, /*batch_size=*/3, /*shuffle=*/false);

    const auto first_epoch = collect_batch_order(loader);
    EXPECT_EQ(first_epoch, (std::vector<size_t>{0u, 1u, 2u, 3u, 4u, 5u}));

    loader.reset();
    const auto second_epoch = collect_batch_order(loader);
    EXPECT_EQ(second_epoch, first_epoch);
}

TEST_F(DataLoaderTest, ShuffleUsesSeedDeterministicallyAcrossEpochs) {
    constexpr size_t n = 12;
    constexpr uint32_t seed = 2026u;

    ToyDataset ds(n, /*input_features=*/3, /*target_features=*/2);
    DataLoader loader(ds, /*batch_size=*/5, /*shuffle=*/true, seed);

    const auto epoch1 = collect_batch_order(loader);
    const auto expected1 = shuffled_indices(n, seed, /*rounds=*/1);
    EXPECT_EQ(epoch1, expected1);

    loader.reset();
    const auto epoch2 = collect_batch_order(loader);
    const auto expected2 = shuffled_indices(n, seed, /*rounds=*/2);
    EXPECT_EQ(epoch2, expected2);

    EXPECT_NE(epoch1, epoch2);
}

TEST_F(DataLoaderTest, ShuffleFalseWithBatchSizeLargerThanDatasetSingleBatch) {
    ToyDataset ds(/*sample_count=*/5, /*input_features=*/3, /*target_features=*/4);
    DataLoader loader(ds, /*batch_size=*/64, /*shuffle=*/false);

    ASSERT_TRUE(loader.has_next());
    auto [x, y] = loader.next_batch();

    EXPECT_EQ(x.shape_at(0), 5u);
    EXPECT_EQ(x.shape_at(1), 3u);
    EXPECT_EQ(y.shape_at(0), 5u);
    EXPECT_EQ(y.shape_at(1), 4u);
    EXPECT_FALSE(loader.has_next());
}

class MNISTDatasetIntegrationTest : public ::testing::Test {};

TEST_F(MNISTDatasetIntegrationTest, FullMNISTTrainSplitLoadsAndMatchesIDXContent) {
    verify_split_with_real_files(
        /*folder=*/"data/MNIST",
        /*split_prefix=*/"train",
        /*expected_rows=*/28,
        /*expected_cols=*/28,
        /*pixel_prefix_to_check=*/64);
}

TEST_F(MNISTDatasetIntegrationTest, Subsample100TrainSplitLoadsAs14x14) {
    verify_split_with_real_files(
        /*folder=*/"data/MNIST_subsamp100",
        /*split_prefix=*/"train",
        /*expected_rows=*/14,
        /*expected_cols=*/14,
        /*pixel_prefix_to_check=*/64);
}

TEST_F(MNISTDatasetIntegrationTest, Subsample1000TrainSplitLoadsAs14x14) {
    verify_split_with_real_files(
        /*folder=*/"data/MNIST_subsamp1000",
        /*split_prefix=*/"train",
        /*expected_rows=*/14,
        /*expected_cols=*/14,
        /*pixel_prefix_to_check=*/64);
}

TEST_F(MNISTDatasetIntegrationTest, Subsample100TestSplitLoadsAs14x14) {
    verify_split_with_real_files(
        /*folder=*/"data/MNIST_subsamp100",
        /*split_prefix=*/"t10k",
        /*expected_rows=*/14,
        /*expected_cols=*/14,
        /*pixel_prefix_to_check=*/64);
}

TEST_F(MNISTDatasetIntegrationTest, DataLoaderBatchesRealSubsampledMNISTCorrectly) {
    const std::string image_path = resolve_data_path("data/MNIST_subsamp1000/train-images-idx3-ubyte");
    const std::string label_path = resolve_data_path("data/MNIST_subsamp1000/train-labels-idx1-ubyte");

    MNISTDataset ds(image_path, label_path);
    DataLoader loader(ds, /*batch_size=*/16, /*shuffle=*/false);

    size_t seen = 0;
    while (loader.has_next()) {
        auto [inputs, targets] = loader.next_batch();

        ASSERT_EQ(inputs.ndim, 2u);
        ASSERT_EQ(targets.ndim, 2u);
        ASSERT_EQ(inputs.shape_at(1), 14u * 14u);
        ASSERT_EQ(targets.shape_at(1), 10u);

        for (size_t b = 0; b < inputs.shape_at(0); ++b) {
            float row_sum = 0.0f;
            for (size_t k = 0; k < 10; ++k) {
                const float t = targets(b, k);
                EXPECT_TRUE(t == 0.0f || t == 1.0f);
                row_sum += t;
            }
            EXPECT_FLOAT_EQ(row_sum, 1.0f);

            for (size_t p = 0; p < inputs.shape_at(1); ++p) {
                const float v = inputs(b, p);
                EXPECT_GE(v, 0.0f);
                EXPECT_LE(v, 1.0f);
            }
        }

        seen += inputs.shape_at(0);
    }

    EXPECT_EQ(seen, ds.size());
}
