#include <gtest/gtest.h>
#include "tensorlib.h"
#include "ops/ops.h"

// ═══════════════════════════════════════════════════════════════
// 1. MulOpForward — shape and value correctness
// ═══════════════════════════════════════════════════════════════

TEST(MulOpForward, OutputShapeMatchesInputs) {
    Tensor A = Tensor::zeros({2,3});
    Tensor B = Tensor::zeros({2,3});

    Tensor C = MulOp::forward(A,B);

    EXPECT_EQ(C.shape(0),2);
    EXPECT_EQ(C.shape(1),3);
}

TEST(MulOpForward, OutputNdimIsTwo) {
    Tensor A = Tensor::zeros({2,2});
    Tensor B = Tensor::zeros({2,2});

    Tensor C = MulOp::forward(A,B);

    EXPECT_EQ(C.ndim(),2);
}

TEST(MulOpForward, ComputesCorrectValues) {
    Tensor A = Tensor::from_data({1,2,3,4},{2,2});
    Tensor B = Tensor::from_data({5,6,7,8},{2,2});

    Tensor C = MulOp::forward(A,B);

    EXPECT_FLOAT_EQ(C.at({0,0}),5.f);
    EXPECT_FLOAT_EQ(C.at({0,1}),12.f);
    EXPECT_FLOAT_EQ(C.at({1,0}),21.f);
    EXPECT_FLOAT_EQ(C.at({1,1}),32.f);
}

TEST(MulOpForward, MultiplyByZeroProducesZero) {
    Tensor A = Tensor::from_data({1,2,3,4},{2,2});
    Tensor Z = Tensor::zeros({2,2});

    Tensor C = MulOp::forward(A,Z);

    for(int r=0;r<2;r++)
        for(int c=0;c<2;c++)
            EXPECT_FLOAT_EQ(C.at({r,c}),0.f);
}

TEST(MulOpForward, MultiplyByOneLeavesTensorUnchanged) {
    Tensor A = Tensor::from_data({1,2,3,4},{2,2});
    Tensor O = Tensor::from_data({1,1,1,1},{2,2});

    Tensor C = MulOp::forward(A,O);

    for(int r=0;r<2;r++)
        for(int c=0;c<2;c++)
            EXPECT_FLOAT_EQ(C.at({r,c}),A.at({r,c}));
}

TEST(MulOpForward, OutputIsContiguous) {
    Tensor A = Tensor::from_data({1,2,3,4},{2,2});
    Tensor B = Tensor::from_data({5,6,7,8},{2,2});

    Tensor C = MulOp::forward(A,B);

    EXPECT_TRUE(C.is_contiguous());
}

// ═══════════════════════════════════════════════════════════════
// 2. Fast path (contiguous)
// ═══════════════════════════════════════════════════════════════

TEST(MulForwardFastPath, BothInputsContiguous) {

    Tensor A = Tensor::from_data({1,2,3,4},{2,2});
    Tensor B = Tensor::from_data({5,6,7,8},{2,2});

    ASSERT_TRUE(A.is_contiguous());
    ASSERT_TRUE(B.is_contiguous());

    Tensor C = MulOp::forward(A,B);

    EXPECT_FLOAT_EQ(C.at({1,1}),32.f);
}

// ═══════════════════════════════════════════════════════════════
// 3. Slow path (non-contiguous)
// ═══════════════════════════════════════════════════════════════

TEST(MulForwardSlowPath, NonContiguousInput) {

    Tensor A = Tensor::from_data({1,3,2,4},{2,2});
    Tensor AT = A.transpose();

    Tensor B = Tensor::from_data({5,6,7,8},{2,2});

    ASSERT_FALSE(AT.is_contiguous());

    Tensor C = MulOp::forward(AT,B);

    EXPECT_FLOAT_EQ(C.at({0,0}),1*5);
    EXPECT_FLOAT_EQ(C.at({1,1}),4*8);
}

// ═══════════════════════════════════════════════════════════════
// 4. Backward values in isolation
// ═══════════════════════════════════════════════════════════════

TEST(MulOpBackward, ReturnsTwoGradients) {

    Tensor A = Tensor::from_data({1,2,3,4},{2,2});
    Tensor B = Tensor::from_data({5,6,7,8},{2,2});
    Tensor grad = Tensor::from_data({1,1,1,1},{2,2});

    auto g = MulOp::backward(grad,A,B,true,true);

    EXPECT_EQ(g.size(),2u);
}

TEST(MulOpBackward, GradientForAIsGradTimesB) {

    Tensor A = Tensor::from_data({1,2,3,4},{2,2});
    Tensor B = Tensor::from_data({5,6,7,8},{2,2});
    Tensor grad = Tensor::from_data({1,1,1,1},{2,2});

    auto g = MulOp::backward(grad,A,B,true,true);

    EXPECT_FLOAT_EQ(g[0].at({0,0}),5.f);
    EXPECT_FLOAT_EQ(g[0].at({0,1}),6.f);
    EXPECT_FLOAT_EQ(g[0].at({1,0}),7.f);
    EXPECT_FLOAT_EQ(g[0].at({1,1}),8.f);
}

TEST(MulOpBackward, GradientForBIsGradTimesA) {

    Tensor A = Tensor::from_data({1,2,3,4},{2,2});
    Tensor B = Tensor::from_data({5,6,7,8},{2,2});
    Tensor grad = Tensor::from_data({1,1,1,1},{2,2});

    auto g = MulOp::backward(grad,A,B,true,true);

    EXPECT_FLOAT_EQ(g[1].at({0,0}),1.f);
    EXPECT_FLOAT_EQ(g[1].at({0,1}),2.f);
    EXPECT_FLOAT_EQ(g[1].at({1,0}),3.f);
    EXPECT_FLOAT_EQ(g[1].at({1,1}),4.f);
}

// ═══════════════════════════════════════════════════════════════
// 5. Autograd wrapper
// ═══════════════════════════════════════════════════════════════

TEST(MulAutograd, OutputRequiresGradIfEitherInputDoes) {

    Tensor A = Tensor::from_data({1,2,3,4},{2,2},{},true);
    Tensor B = Tensor::from_data({5,6,7,8},{2,2},{},false);

    Tensor C = mul(A,B);

    EXPECT_TRUE(C.requires_grad());
}

TEST(MulAutograd, OutputDoesNotRequireGradIfNeitherDoes) {

    Tensor A = Tensor::from_data({1,2,3,4},{2,2});
    Tensor B = Tensor::from_data({5,6,7,8},{2,2});

    Tensor C = mul(A,B);

    EXPECT_FALSE(C.requires_grad());
}

// ═══════════════════════════════════════════════════════════════
// 6. Full backward through graph
// ═══════════════════════════════════════════════════════════════

TEST(MulBackwardValues, GradientAccumulatesOnInputs) {

    Tensor A = Tensor::from_data({1,2,3,4},{2,2},{},true);
    Tensor B = Tensor::from_data({5,6,7,8},{2,2},{},true);

    Tensor C = mul(A,B);

    backward(C);

    ASSERT_TRUE(A.has_grad());
    ASSERT_TRUE(B.has_grad());

    EXPECT_FLOAT_EQ(A.grad().at({0,0}),5.f);
    EXPECT_FLOAT_EQ(B.grad().at({0,0}),1.f);
}

TEST(MulBackwardValues, GradientShapeMatchesInputs) {

    Tensor A = Tensor::from_data({1,2,3,4},{2,2},{},true);
    Tensor B = Tensor::from_data({5,6,7,8},{2,2},{},true);

    Tensor C = mul(A,B);

    backward(C);

    EXPECT_EQ(A.grad().shape(0),2);
    EXPECT_EQ(A.grad().shape(1),2);
}