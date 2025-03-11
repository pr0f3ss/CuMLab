#ifndef CUMLAB_MAE_LOSS_HPP
#define CUMLAB_MAE_LOSS_HPP

#include "CuMLab/loss/loss.hpp"
#include <cmath>
#include <memory>

namespace CuMLab {

/**
 * @brief Mean Absolute Error (L1 Loss).
 *
 * Computes `MAE = mean(|prediction - target|)`
 */
template <typename T> class MAELoss : public Loss<T> {
public:
  MAELoss() = default;

  /**
   * @brief Computes the MAE loss.
   */
  std::shared_ptr<Tensor<T>>
  compute_loss(const std::shared_ptr<Tensor<T>> &prediction,
               const std::shared_ptr<Tensor<T>> &target) override {

    if (prediction->shape() != target->shape()) {
      throw std::invalid_argument(
          "MAE Loss: Shape mismatch between prediction and target.");
    }

    auto diff = (*prediction) - (*target);
    auto abs_diff = std::make_shared<Tensor<T>>(diff.shape());

    for (size_t i = 0; i < diff.size(); ++i) {
      (*abs_diff)({static_cast<int>(i)}) =
          std::abs(diff({static_cast<int>(i)}));
    }

    T mae_value = abs_diff->sum() / static_cast<T>(prediction->size());

    return std::make_shared<Tensor<T>>(std::vector<int>{1},
                                       std::vector<T>{mae_value});
  }
};

} // namespace CuMLab

#endif // CUMLAB_MAE_LOSS_HPP
