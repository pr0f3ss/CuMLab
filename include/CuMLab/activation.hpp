#ifndef CUMLAB_ACTIVATION_HPP
#define CUMLAB_ACTIVATION_HPP

#include "CuMLab/module.hpp"
#include "CuMLab/tensor.hpp"
#include <cmath>
#include <memory>

namespace CuMLab {

/**
 * @brief Base class for activation functions.
 */
template <typename T> class Activation : public Module<T> {
public:
  Activation() = default;
  virtual ~Activation() = default;

  /**
   * @brief Forward pass for activation function.
   *
   * @param input The input tensor.
   * @return The output tensor after activation.
   */
  virtual std::shared_ptr<Tensor<T>>
  forward(const std::shared_ptr<Tensor<T>> &input) override = 0;
};

} // namespace CuMLab

#endif // CUMLAB_ACTIVATION_HPP
