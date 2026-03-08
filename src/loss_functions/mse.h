
#include "tensorlib.h"
#include "ops/ops.h"


inline Tensor mse_loss(const Tensor& pred, const Tensor& target) {

    assert(pred.ndim() == 2 && target.ndim() == 2);
    assert(pred.shape(0) == target.shape(0));
    assert(pred.shape(1) == target.shape(1));

    int64_t rows = pred.shape(0);
    int64_t cols = pred.shape(1);
    float n = (float)(rows * cols);

    float sum = 0.f;
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t c = 0; c < cols; c++) {
            float diff = pred.at({r, c}) - target.at({r, c});
            sum += diff * diff;
        }
    }

    // result is a single number stored in a [1,1] tensor
    Tensor loss = Tensor::from_data({sum / n}, {1, 1});

    if (pred.requires_grad()) {

        std::shared_ptr<AutogradMeta> autograd_meta_pred = pred.autograd_meta;

        Tensor saved_pred   = pred.clone();
        Tensor saved_target = target.clone();

        std::function<std::vector<Tensor>(const Tensor&)> backward_fn =
            [saved_pred, saved_target, rows, cols, n]
            (const Tensor& grad) -> std::vector<Tensor> {

                // grad is [1,1] — a single upstream scalar
                float upstream = grad.at({0, 0});

                Tensor dpred = Tensor::zeros({rows, cols});

                for (int64_t r = 0; r < rows; r++) {
                    for (int64_t c = 0; c < cols; c++) {
                        float diff = saved_pred.at({r, c}) - saved_target.at({r, c});
                        dpred.at({r, c}) = upstream * 2.f * diff / n;
                    }
                }

                return {dpred};
            };

        loss.autograd_meta = make_grad_meta("mse_loss", {autograd_meta_pred}, backward_fn);
    }

    return loss;
}