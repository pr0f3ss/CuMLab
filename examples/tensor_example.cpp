#include "CuMLab/core/tensor.hpp"
#include <iostream>

int main() {
  using namespace CuMLab;

  auto A = std::make_shared<Tensor<float>>(std::vector<int>{2, 3}, true);
  auto B = std::make_shared<Tensor<float>>(std::vector<int>{2, 3}, true);

  // Assign values
  (*A)({0, 0}) = 1.0f;
  (*A)({0, 1}) = 2.0f;
  (*A)({0, 2}) = 3.0f;
  (*A)({1, 0}) = 4.0f;
  (*A)({1, 1}) = 5.0f;
  (*A)({1, 2}) = 6.0f;

  (*B)({0, 0}) = 6.0f;
  (*B)({0, 1}) = 5.0f;
  (*B)({0, 2}) = 4.0f;
  (*B)({1, 0}) = 3.0f;
  (*B)({1, 1}) = 2.0f;
  (*B)({1, 2}) = 1.0f;

  std::cout << "A:" << std::endl;
  A->print();

  std::cout << "B:" << std::endl;
  B->print();

  std::cout << "A+B:" << std::endl;
  (A + B)->print();

  std::cout << "A-B:" << std::endl;
  (A - B)->print();

  std::cout << "A*B^T:" << std::endl;
  (matmul(A, transpose(B)))->print();

  std::cout << "A*B:" << std::endl;
  // (A * B).print();

  std::cout << "A/B:" << std::endl;
  // (A / B)->print();

  std::cout << "-A:" << std::endl;
  // (-A).print();

  std::cout << "Sum: " << A->sum() << std::endl;
  std::cout << "Mean: " << A->mean() << std::endl;
  std::cout << "Max: " << A->max() << std::endl;
  std::cout << "Min: " << A->min() << std::endl;

  return 0;
}
