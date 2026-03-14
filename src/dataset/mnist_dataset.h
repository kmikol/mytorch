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

    size_t size() const override;
    Sample get(size_t index) const override;

    size_t input_size() const { return static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols); }
    size_t num_classes() const { return 10u; }
    uint32_t image_rows() const { return num_rows; }
    uint32_t image_cols() const { return num_cols; }

private:
    static uint32_t read_be_uint32(std::ifstream& file);

    std::vector<std::vector<float>> images;
    std::vector<uint8_t> labels;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;
};
