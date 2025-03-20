#ifndef CUMLAB_LINEAR_HPP
#define CUMLAB_LINEAR_HPP

#include "CuMLab/core/tensor.hpp"
#include "CuMLab/layers/module.hpp"
#include <limits>
#include <memory>
#include <random>
#include <type_traits>

namespace CuMLab {

/**
 * @brief Fully Connected (Linear) Layer.
 *
 * This layer performs the operation Y = XW + B, where:
 * - X is the input tensor
 * - W is the weight matrix
 * - B is the bias vector
 */
template <typename T> class Linear : public Module<T> {
private:
  int in_features_, out_features_;

public:
  /**
   * @brief Constructs a linear layer.
   *
   * @param in_features The number of input features.
   * @param out_features The number of output features.
   */
  Linear(int in_features, int out_features)
      : in_features_(in_features), out_features_(out_features) {

    auto weight = std::make_shared<Tensor<T>>(
        std::vector<int>{in_features, out_features});
    auto bias = std::make_shared<Tensor<T>>(std::vector<int>{out_features});

    // Initialize weights based on type T
    if constexpr (std::is_floating_point<T>::value) {
      // Use uniform random distribution for floating point types (float,
      // double)
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<T> dist(static_cast<T>(-0.1),
                                             static_cast<T>(0.1));

      for (int i = 0; i < weight->size(); ++i) {
        (*weight)({i}) = dist(gen);
      }
    } else {
      // Integer types: Use small values between -1 and 1
      for (int i = 0; i < weight->size(); ++i) {
        (*weight)({i}) = static_cast<T>(
            i % 2 == 0 ? 1 : -1); // Alternate 1, -1 for stability
      }
    }

    // Register weight and bias parameters
    this->register_parameter("weight", weight);
    this->register_parameter("bias", bias);
  }

  /**
   * @brief Forward pass of the linear layer.
   *
   * Computes Y = XW + B.
   *
   * @param input The input tensor.
   * @return The output tensor after applying the linear transformation.
   */
  std::shared_ptr<Tensor<T>>
  forward(const std::shared_ptr<Tensor<T>> &input) override {
    auto weight = this->parameters_["weight"];
    auto bias = this->parameters_["bias"];

    auto output = input * weight + bias;

    return output;
  }
};

} // namespace CuMLab

#endif // CUMLAB_LINEAR_HPP
