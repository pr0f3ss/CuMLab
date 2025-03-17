#include "CuMLab/core/tensor.hpp"
#include <iostream>

int main() {
  using namespace CuMLab;

  // Create tensors with `requires_grad = true`
  auto A = std::make_shared<Tensor<float>>(std::vector<int>{2, 3}, true);
  auto B = std::make_shared<Tensor<float>>(std::vector<int>{3, 2}, true);
  auto bias = std::make_shared<Tensor<float>>(std::vector<int>{2}, true);

  // Assign values
  (*A)({0, 0}) = 1.0f;
  (*A)({0, 1}) = 2.0f;
  (*A)({0, 2}) = 3.0f;
  (*A)({1, 0}) = 4.0f;
  (*A)({1, 1}) = 5.0f;
  (*A)({1, 2}) = 6.0f;

  (*B)({0, 0}) = 1.0f;
  (*B)({0, 1}) = 2.0f;
  (*B)({1, 0}) = 3.0f;
  (*B)({1, 1}) = 4.0f;
  (*B)({2, 0}) = 5.0f;
  (*B)({2, 1}) = 6.0f;

  (*bias)({0}) = 0.5f;
  (*bias)({1}) = -0.5f;

  // Compute function: Z = A * B + bias (broadcasted)
  auto Z = matmul(A, B) + bias;
  std::cout << "Z = A * B + bias:\n";
  Z->print();

  // Compute gradients
  Z->backward();

  // Print gradients
  std::cout << "Gradient of A:\n";
  A->grad()->print();

  std::cout << "Gradient of B:\n";
  B->grad()->print();

  std::cout << "Gradient of bias:\n";
  bias->grad()->print();

  return 0;
}
