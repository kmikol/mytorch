#include "dataset/mnist_dataset.h"

#include <cassert>
#include <fstream>


static Shape make_shape_2d(size_t d0, size_t d1) {
    Shape s{};
    s[0] = d0;
    s[1] = d1;
    return s;
}


uint32_t MNISTDataset::read_be_uint32(std::ifstream& file) {
    unsigned char bytes[4]{};
    file.read(reinterpret_cast<char*>(bytes), 4);
    assert(file.good());

    return (static_cast<uint32_t>(bytes[0]) << 24)
         | (static_cast<uint32_t>(bytes[1]) << 16)
         | (static_cast<uint32_t>(bytes[2]) << 8)
         |  static_cast<uint32_t>(bytes[3]);
}


MNISTDataset::MNISTDataset(const std::string& image_file,
                           const std::string& label_file) {
    std::ifstream img(image_file, std::ios::binary);
    std::ifstream lbl(label_file, std::ios::binary);
    assert(img.is_open() && lbl.is_open());

    uint32_t img_magic = read_be_uint32(img);
    uint32_t img_count = read_be_uint32(img);
    num_rows = read_be_uint32(img);
    num_cols = read_be_uint32(img);

    uint32_t lbl_magic = read_be_uint32(lbl);
    uint32_t lbl_count = read_be_uint32(lbl);

    assert(img_magic == 2051u);
    assert(lbl_magic == 2049u);
    assert(img_count == lbl_count);

    images.resize(img_count);
    labels.resize(img_count);

    size_t pixels_per_image = static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols);

    for (uint32_t i = 0; i < img_count; ++i) {
        images[i].resize(pixels_per_image);
        for (size_t p = 0; p < pixels_per_image; ++p) {
            unsigned char pixel = 0;
            img.read(reinterpret_cast<char*>(&pixel), 1);
            assert(img.good());
            images[i][p] = static_cast<float>(pixel) / 255.f;
        }

        lbl.read(reinterpret_cast<char*>(&labels[i]), 1);
        assert(lbl.good());
        assert(labels[i] < 10);
    }
}


size_t MNISTDataset::size() const {
    return images.size();
}


Sample MNISTDataset::get(size_t index) const {
    assert(index < images.size());

    size_t pixels_per_image = static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols);

    Tensor input = Tensor::zeros(make_shape_2d(1, pixels_per_image), 2);
    for (size_t p = 0; p < pixels_per_image; ++p)
        input(0, p) = images[index][p];

    Tensor target = Tensor::zeros(make_shape_2d(1, 10), 2);
    target(0, labels[index]) = 1.f;

    return {input, target};
}
