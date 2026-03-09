#include <gtest/gtest.h>
#include <vector>
#include "tensorlib.h"
#include "ops/ops.h"

// ═══════════════════════════════════════════════════════════════
// 1. AddOpForward — shape and value correctness
// ═══════════════════════════════════════════════════════════════

TEST(AddOpForward, OutputShapeMatchesInputs) {
    Tensor A = Tensor::zeros({2,3});
    Tensor B = Tensor::zeros({2,3});

    Tensor C = AddOp::forward(A,B);

    EXPECT_EQ(C.shape(0),2);
    EXPECT_EQ(C.shape(1),3);
}

TEST(AddOpForward, OutputNdimIsTwo) {
    Tensor A = Tensor::zeros({2,2});
    Tensor B = Tensor::zeros({2,2});

    Tensor C = AddOp::forward(A,B);

    EXPECT_EQ(C.ndim(),2);
}

TEST(AddOpForward, ComputesCorrectValues) {
    Tensor A = Tensor::from_data({1,2,3,4},{2,2});
    Tensor B = Tensor::from_data({5,6,7,8},{2,2});

    Tensor C = AddOp::forward(A,B);

    EXPECT_FLOAT_EQ(C.at(0,0),6.f);
    EXPECT_FLOAT_EQ(C.at(0,1),8.f);
    EXPECT_FLOAT_EQ(C.at(1,0),10.f);
    EXPECT_FLOAT_EQ(C.at(1,1),12.f);
}

TEST(AddOpForward, AddZeroLeavesTensorUnchanged) {
    Tensor A = Tensor::from_data({1,2,3,4},{2,2});
    Tensor Z = Tensor::zeros({2,2});

    Tensor C = AddOp::forward(A,Z);

    for(int r=0;r<2;r++)
        for(int c=0;c<2;c++)
            EXPECT_FLOAT_EQ(C.at(r,c),A.at(r,c));
}

TEST(AddOpForward, OutputIsContiguous) {
    Tensor A = Tensor::from_data({1,2,3,4},{2,2});
    Tensor B = Tensor::from_data({5,6,7,8},{2,2});

    Tensor C = AddOp::forward(A,B);

    EXPECT_TRUE(C.is_contiguous());
}

// ═══════════════════════════════════════════════════════════════
// 2. Broadcasting behaviour
// ═══════════════════════════════════════════════════════════════

TEST(AddBroadcast, BroadcastSecondTensorAcrossColumns) {

    Tensor A = Tensor::from_data({
        1,2,3,
        4,5,6
    }, {2,3});

    Tensor B = Tensor::from_data({
        10,
        20
    }, {2,1});

    Tensor C = AddOp::forward(A,B);

    EXPECT_FLOAT_EQ(C.at(0,0),11);
    EXPECT_FLOAT_EQ(C.at(0,1),12);
    EXPECT_FLOAT_EQ(C.at(0,2),13);

    EXPECT_FLOAT_EQ(C.at(1,0),24);
    EXPECT_FLOAT_EQ(C.at(1,1),25);
    EXPECT_FLOAT_EQ(C.at(1,2),26);
}

TEST(AddBroadcast, BroadcastFirstTensorAcrossColumns) {

    Tensor A = Tensor::from_data({
        1,
        2
    }, {2,1});

    Tensor B = Tensor::from_data({
        10,20,30,
        40,50,60
    }, {2,3});

    Tensor C = AddOp::forward(A,B);

    EXPECT_FLOAT_EQ(C.at(0,0),11);
    EXPECT_FLOAT_EQ(C.at(0,1),21);
    EXPECT_FLOAT_EQ(C.at(0,2),31);

    EXPECT_FLOAT_EQ(C.at(1,0),42);
    EXPECT_FLOAT_EQ(C.at(1,1),52);
    EXPECT_FLOAT_EQ(C.at(1,2),62);
}

// ═══════════════════════════════════════════════════════════════
// 3. AddOpBackward — gradients in isolation
// ═══════════════════════════════════════════════════════════════

TEST(AddOpBackward, ReturnsTwoGradients) {

    Tensor grad = Tensor::from_data({1,1,1,1},{2,2});

    AddOp::BroadcastInfo info{false,false};

    auto g = AddOp::backward(grad,true,true,info,2,2);

    EXPECT_EQ(g.size(),2u);
}

TEST(AddOpBackward, GradientPassesThroughWithoutBroadcast) {

    Tensor grad = Tensor::from_data({1,2,3,4},{2,2});

    AddOp::BroadcastInfo info{false,false};

    auto g = AddOp::backward(grad,true,true,info,2,2);

    EXPECT_FLOAT_EQ(g[0].at(0,0),1);
    EXPECT_FLOAT_EQ(g[0].at(1,1),4);

    EXPECT_FLOAT_EQ(g[1].at(0,0),1);
    EXPECT_FLOAT_EQ(g[1].at(1,1),4);
}

TEST(AddOpBackward, BroadcastGradientForBIsReduced) {

    Tensor grad = Tensor::from_data({
        1,1,1,
        1,1,1
    },{2,3});

    AddOp::BroadcastInfo info;
    info.b_broadcast_cols = true;
    info.a_broadcast_cols = false;

    auto g = AddOp::backward(grad,true,true,info,3,1);

    EXPECT_FLOAT_EQ(g[1].at(0,0),3);
    EXPECT_FLOAT_EQ(g[1].at(1,0),3);
}

TEST(AddOpBackward, BroadcastGradientForAIsReduced) {

    Tensor grad = Tensor::from_data({
        1,1,1,
        2,2,2
    },{2,3});

    AddOp::BroadcastInfo info;
    info.a_broadcast_cols = true;
    info.b_broadcast_cols = false;

    auto g = AddOp::backward(grad,true,true,info,1,3);

    EXPECT_FLOAT_EQ(g[0].at(0,0),3);
    EXPECT_FLOAT_EQ(g[0].at(1,0),6);
}

// ═══════════════════════════════════════════════════════════════
// 4. Autograd wrapper behaviour
// ═══════════════════════════════════════════════════════════════

TEST(AddAutograd, OutputRequiresGradIfEitherInputDoes) {

    Tensor A = Tensor::from_data({1,2,3,4},{2,2},{},true);
    Tensor B = Tensor::from_data({5,6,7,8},{2,2},{},false);

    Tensor C = add(A,B);

    EXPECT_TRUE(C.requires_grad());
}

TEST(AddAutograd, OutputDoesNotRequireGradIfNeitherInputDoes) {

    Tensor A = Tensor::from_data({1,2,3,4},{2,2});
    Tensor B = Tensor::from_data({5,6,7,8},{2,2});

    Tensor C = add(A,B);

    EXPECT_FALSE(C.requires_grad());
}

TEST(AddAutograd, OutputHasNoGradBeforeBackward) {

    Tensor A = Tensor::from_data({1,2,3,4},{2,2},{},true);
    Tensor B = Tensor::from_data({5,6,7,8},{2,2},{},true);

    Tensor C = add(A,B);

    EXPECT_FALSE(C.has_grad());
}

// ═══════════════════════════════════════════════════════════════
// 5. Full backward pass
// ═══════════════════════════════════════════════════════════════

TEST(AddBackwardValues, GradientAccumulatesOnInputs) {

    Tensor A = Tensor::from_data({1,2,3,4},{2,2},{},true);
    Tensor B = Tensor::from_data({5,6,7,8},{2,2},{},true);

    Tensor C = add(A,B);

    backward(C);

    ASSERT_TRUE(A.has_grad());
    ASSERT_TRUE(B.has_grad());

    EXPECT_FLOAT_EQ(A.grad().at(0,0),1);
    EXPECT_FLOAT_EQ(A.grad().at(1,1),1);

    EXPECT_FLOAT_EQ(B.grad().at(0,0),1);
    EXPECT_FLOAT_EQ(B.grad().at(1,1),1);
}

TEST(AddBackwardValues, BroadcastGradientAccumulatesCorrectly) {

    Tensor A = Tensor::from_data({
        1,2,3,
        4,5,6
    },{2,3},{},true);

    Tensor B = Tensor::from_data({
        10,
        20
    },{2,1},{},true);

    Tensor C = add(A,B);

    backward(C);

    ASSERT_TRUE(B.has_grad());

    EXPECT_FLOAT_EQ(B.grad().at(0,0),3);
    EXPECT_FLOAT_EQ(B.grad().at(1,0),3);
}