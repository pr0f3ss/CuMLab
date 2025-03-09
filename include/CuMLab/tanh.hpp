#ifndef CUMLAB_TANH_HPP
#define CUMLAB_TANH_HPP

#include "CuMLab/activation.hpp"
#include <cmath>
#include <memory>

namespace CuMLab {

/**
 * @brief Hyperbolic Tangent (Tanh) activation function.
 */
template <typename T> class Tanh : public Activation<T> {
public:
  Tanh() = default;

  /**
   * @brief Applies Tanh function element-wise.
   */
  std::shared_ptr<Tensor<T>>
  forward(const std::shared_ptr<Tensor<T>> &input) override {
    auto output = std::make_shared<Tensor<T>>(input->shape());
    for (size_t i = 0; i < input->size(); ++i) {
      output->data()[i] = std::tanh(input->data()[i]);
    }
    return output;
  }
};

} // namespace CuMLab

#endif // CUMLAB_TANH_HPP
