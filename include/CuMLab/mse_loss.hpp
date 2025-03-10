#ifndef CUMLAB_MSE_LOSS_HPP
#define CUMLAB_MSE_LOSS_HPP

#include "CuMLab/loss.hpp"
#include <memory>

namespace CuMLab {

/**
 * @brief Mean Squared Error (MSE) Loss.
 *
 * Computes `MSE = mean((prediction - target)²)`
 */
template <typename T> class MSELoss : public Loss<T> {
public:
  MSELoss() = default;

  /**
   * @brief Computes the MSE loss.
   */
  std::shared_ptr<Tensor<T>>
  compute_loss(const std::shared_ptr<Tensor<T>> &prediction,
               const std::shared_ptr<Tensor<T>> &target) override {

    if (prediction->shape() != target->shape()) {
      throw std::invalid_argument(
          "MSE Loss: Shape mismatch between prediction and target.");
    }

    auto diff = (*prediction) - (*target);
    auto squared_diff = diff * diff;
    T mse_value = squared_diff.sum() / static_cast<T>(prediction->size());

    return std::make_shared<Tensor<T>>(std::vector<int>{1},
                                       std::vector<T>{mse_value});
  }
};

} // namespace CuMLab

#endif // CUMLAB_MSE_LOSS_HPP
