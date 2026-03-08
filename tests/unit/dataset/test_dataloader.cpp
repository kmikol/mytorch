// tests/test_dataloader.cpp

#include <gtest/gtest.h>
#include "dataset/dataloader.h"

class DummyDataset : public Dataset {
public:

    size_t size() const override { return 8; }

    Sample get(size_t index) const override
    {
        Tensor x = Tensor::from_data({(float)index}, {1,1});
        Tensor y = Tensor::from_data({(float)(index*2)}, {1,1});
        return {x,y};
    }
};

TEST(DataLoaderTest, BatchShapeCorrect)
{
    DummyDataset dataset;
    DataLoader loader(dataset,4,false);

    auto [x,y] = loader.next_batch();

    EXPECT_EQ(x.shape(0),1);
    EXPECT_EQ(x.shape(1),4);

    EXPECT_EQ(y.shape(0),1);
    EXPECT_EQ(y.shape(1),4);
}

TEST(DataLoaderTest, IteratesEntireDataset)
{
    DummyDataset dataset;
    DataLoader loader(dataset,2,false);

    int batches = 0;

    while(loader.has_next())
    {
        loader.next_batch();
        batches++;
    }

    EXPECT_EQ(batches,4);
}

TEST(DataLoaderTest, ResetRestartsEpoch)
{
    DummyDataset dataset;
    DataLoader loader(dataset,4,false);

    loader.next_batch();
    loader.next_batch();

    EXPECT_FALSE(loader.has_next());

    loader.reset();

    EXPECT_TRUE(loader.has_next());
}