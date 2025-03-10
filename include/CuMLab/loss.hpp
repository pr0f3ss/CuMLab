#ifndef CUMLAB_LOSS_HPP
#define CUMLAB_LOSS_HPP

#include "CuMLab/tensor.hpp"
#include <memory>

namespace CuMLab {

/**
 * @brief Base class for loss functions.
 */
template <typename T> class Loss {
public:
  Loss() = default;
  virtual ~Loss() = default;

  /**
   * @brief Computes the loss between prediction and target.
   * @param prediction The model's output tensor.
   * @param target The ground truth tensor.
   * @return The computed loss (scalar tensor).
   */
  virtual std::shared_ptr<Tensor<T>>
  compute_loss(const std::shared_ptr<Tensor<T>> &prediction,
               const std::shared_ptr<Tensor<T>> &target) = 0;
};

} // namespace CuMLab

#endif // CUMLAB_LOSS_HPP
