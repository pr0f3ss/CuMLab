#ifndef CUMLAB_LINEAR_HPP
#define CUMLAB_LINEAR_HPP

#include "CuMLab/module.hpp"
#include "CuMLab/tensor.hpp"

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

    // Register weight and bias parameters
    this->register_parameter("weight",
                             std::make_shared<Tensor<T>>(
                                 std::vector<int>{in_features, out_features}));

    this->register_parameter(
        "bias", std::make_shared<Tensor<T>>(std::vector<int>{out_features}));
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

    auto output = std::make_shared<Tensor<T>>(*input * (*weight));
    *output = *output + (*bias);

    return output;
  }
};

} // namespace CuMLab

#endif // CUMLAB_LINEAR_HPP
