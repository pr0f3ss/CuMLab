#include "CuMLab/tensor.hpp"
#include <iostream>


int main() {
  Tensor t({2, 3}); // A 2x3 tensor

  // Assign some values
  t({0, 0}) = 1.0f;
  t({0, 1}) = 2.0f;
  t({1, 2}) = 3.5f;

  std::cout << "Original Tensor:" << std::endl;
  t.print();

  // Perform addition
  Tensor t2({2, 3});
  t2({0, 0}) = 0.5f;
  t2({1, 2}) = 1.5f;

  Tensor sum = t + t2;
  std::cout << "Sum of Tensors:" << std::endl;
  sum.print();

  // Scalar multiplication
  Tensor scaled = t * 2.0f;
  std::cout << "Scaled Tensor:" << std::endl;
  scaled.print();

  return 0;
}
