#ifndef CUMLAB_RELU_HPP
#define CUMLAB_RELU_HPP

#include "CuMLab/activation.hpp"
#include <memory>

namespace CuMLab {

/**
 * @brief Rectified Linear Unit (ReLU) activation function.
 */
template <typename T> class ReLU : public Activation<T> {
public:
  ReLU() = default;

  /**
   * @brief Applies ReLU: `f(x) = max(0, x)`
   */
  std::shared_ptr<Tensor<T>>
  forward(const std::shared_ptr<Tensor<T>> &input) override {
    auto output = std::make_shared<Tensor<T>>(input->shape());
    for (size_t i = 0; i < static_cast<size_t>(input->size()); ++i) {
      (*output)({static_cast<int>(i)}) =
          std::max(static_cast<T>(0), (*input)({static_cast<int>(i)}));
    }
    return output;
  }
};

} // namespace CuMLab

#endif // CUMLAB_RELU_HPP
