#ifndef CUMLAB_SIGMOID_HPP
#define CUMLAB_SIGMOID_HPP

#include "CuMLab/layers/activation.hpp"
#include <cmath>
#include <memory>

namespace CuMLab {

/**
 * @brief Sigmoid activation function: `sigma(x) = 1 / (1 + exp(-x))`
 */
template <typename T> class Sigmoid : public Activation<T> {
public:
  Sigmoid() = default;

  /**
   * @brief Applies Sigmoid function element-wise.
   */
  std::shared_ptr<Tensor<T>>
  forward(const std::shared_ptr<Tensor<T>> &input) override {
    auto output = std::make_shared<Tensor<T>>(input->shape());
    for (size_t i = 0; i < static_cast<size_t>(input->size()); ++i) {
      (*output)({static_cast<int>(i)}) =
          1 / (1 + std::exp(-(*input)({static_cast<int>(i)})));
    }
    return output;
  }
};

} // namespace CuMLab

#endif // CUMLAB_SIGMOID_HPP
