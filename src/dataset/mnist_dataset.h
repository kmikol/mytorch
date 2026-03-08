// dataset/mnist_dataset.h

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <cstdint>

#include "dataset.h"

class MNISTDataset : public Dataset {
public:

    MNISTDataset(const std::string& image_file,
                 const std::string& label_file);

    size_t size() const override;

    Sample get(size_t index) const override;

private:

    std::vector<std::vector<float>> images;
    std::vector<uint8_t> labels;

    static uint32_t read_be_uint32(std::ifstream& file);
};