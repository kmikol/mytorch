#include "mnist_dataset.h"

uint32_t MNISTDataset::read_be_uint32(std::ifstream& file)
{
    uint32_t v;
    file.read(reinterpret_cast<char*>(&v), 4);
    return __builtin_bswap32(v);
}

MNISTDataset::MNISTDataset(const std::string& image_file,
                           const std::string& label_file)
{
    std::ifstream img(image_file, std::ios::binary);
    std::ifstream lbl(label_file, std::ios::binary);

    uint32_t magic  = read_be_uint32(img);
    uint32_t count  = read_be_uint32(img);
    uint32_t rows   = read_be_uint32(img);
    uint32_t cols   = read_be_uint32(img);

    read_be_uint32(lbl);
    read_be_uint32(lbl);

    images.resize(count);
    labels.resize(count);

    for (uint32_t i = 0; i < count; ++i) {

        images[i].resize(rows * cols);

        for (uint32_t p = 0; p < rows*cols; ++p) {
            unsigned char pixel;
            img.read((char*)&pixel,1);
            images[i][p] = pixel / 255.f;
        }

        lbl.read((char*)&labels[i],1);
    }
}

size_t MNISTDataset::size() const
{
    return images.size();
}

Sample MNISTDataset::get(size_t index) const
{
    Tensor input = Tensor::from_data(images[index], {784,1});

    std::vector<float> onehot(10,0.f);
    onehot[labels[index]] = 1.f;

    Tensor target = Tensor::from_data(onehot,{10,1});

    return {input,target};
}