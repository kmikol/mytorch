#pragma once
#include "tensorlib.h"

struct AddOp {

    // describes how B was broadcast relative to A
    // captured at forward time, used in backward
    struct BroadcastInfo {
        bool b_broadcast_cols;  // B had 1 col, A had N cols
        bool a_broadcast_cols;  // A had 1 col, B had N cols
        // rows must always match — we don't broadcast rows
    };

    static BroadcastInfo get_broadcast_info(const Tensor& A, const Tensor& B) {
        assert(A.shape(0) == B.shape(0) &&
               "add: row count must match");
        assert((A.shape(1) == B.shape(1) ||
                A.shape(1) == 1          ||
                B.shape(1) == 1) &&
               "add: columns must match or one must be 1");

        BroadcastInfo info;
        info.b_broadcast_cols = (B.shape(1) == 1 && A.shape(1) > 1);
        info.a_broadcast_cols = (A.shape(1) == 1 && B.shape(1) > 1);
        return info;
    }

    static Tensor forward(const Tensor& A, const Tensor& B) {
        BroadcastInfo info = get_broadcast_info(A, B);

        // output shape is the larger of the two
        int64_t rows = A.shape(0);
        int64_t cols = std::max(A.shape(1), B.shape(1));

        Tensor C = Tensor::zeros({rows, cols});

        for (int64_t r = 0; r < rows; r++) {
            for (int64_t c = 0; c < cols; c++) {
                // if A or B is broadcast, always read from column 0
                int64_t ac = info.a_broadcast_cols ? 0 : c;
                int64_t bc = info.b_broadcast_cols ? 0 : c;
                C.at(r, c) = A.at(r, ac) + B.at(r, bc);
            }
        }

        return C;
    }

    static std::vector<Tensor> backward(
        const Tensor& grad,
        bool rA, bool rB,
        BroadcastInfo info,
        int64_t A_cols, int64_t B_cols)
    {

        int64_t rows = grad.shape(0);
        int64_t cols = grad.shape(1);

        std::vector<Tensor> grads(2);

        if (rA) {
            if (info.a_broadcast_cols) {
                // A was broadcast — sum gradient back to [rows, 1]
                Tensor dA = Tensor::zeros({rows, 1});
                for (int64_t r = 0; r < rows; r++)
                    for (int64_t c = 0; c < cols; c++)
                        dA.at(r, 0) += grad.at(r, c);
                grads[0] = dA;
            } else {
                // no broadcast — gradient passes straight through
                grads[0] = grad.clone();
            }
        }

        if (rB) {
            if (info.b_broadcast_cols) {
                // B was broadcast — sum gradient back to [rows, 1]
                // each column of grad contributed to the same B column
                // so we sum them all together
                Tensor dB = Tensor::zeros({rows, 1});
                for (int64_t r = 0; r < rows; r++)
                    for (int64_t c = 0; c < cols; c++)
                        dB.at(r, 0) += grad.at(r, c);
                grads[1] = dB;
            } else {
                grads[1] = grad.clone();
            }
        }

        return grads;
    }
};

inline Tensor add(const Tensor& A, const Tensor& B) {
    assert(A.ndim() == 2 && B.ndim() == 2);

    AddOp::BroadcastInfo info = AddOp::get_broadcast_info(A, B);
    Tensor C = AddOp::forward(A, B);

    bool rA = A.requires_grad(), rB = B.requires_grad();
    if (grad_mode_enabled && (rA || rB)) {
        NoGradGuard no_grad;

        // no tensors need saving — add backward only needs
        // the broadcast info and column counts, captured by value
        int64_t A_cols = A.shape(1), B_cols = B.shape(1);

        C.autograd_meta = make_grad_meta(
            "add",
            {A.autograd_meta, B.autograd_meta},
            [rA, rB, info, A_cols, B_cols](const Tensor& grad) {
                return AddOp::backward(grad, rA, rB, info, A_cols, B_cols);
            });
    }

    return C;
}