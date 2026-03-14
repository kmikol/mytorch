#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "dataset/dataset.h"


class MNISTDataset : public Dataset {
public:
    MNISTDataset(const std::string& image_file, const std::string& label_file);

    size_t size()       const override;
    Sample get(size_t index) const override;

    size_t input_dim()  const override { return static_cast<size_t>(num_rows) * num_cols; }
    size_t target_dim() const override { return 10u; }

    void fill_sample(size_t index, float* input_buf, float* target_buf) const override;

    uint32_t image_rows() const { return num_rows; }
    uint32_t image_cols() const { return num_cols; }

private:
    static uint32_t read_be_uint32(std::ifstream& file);

    std::vector<std::vector<float>> images;
    std::vector<uint8_t> labels;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;
};
