#include "CuMLab/tensor.hpp"
#include <iostream>

int main() {
  using namespace CuMLab;

  Tensor<float> t1({2, 3});
  Tensor<float> t2({2, 3});

  // Assign values
  t1({0, 0}) = 2.0f;
  t1({0, 1}) = 4.0f;
  t1({1, 2}) = 6.0f;

  t2({0, 0}) = 1.0f;
  t2({0, 1}) = 2.0f;
  t2({0, 2}) = 3.0f;
  t2({1, 0}) = 4.0f;
  t2({1, 1}) = 5.0f;
  t2({1, 2}) = 6.0f;

  std::cout << "Tensor 1:" << std::endl;
  t1.print();

  std::cout << "Tensor 2:" << std::endl;
  t2.print();

  std::cout << "Addition:" << std::endl;
  (t1 + t2).print();

  std::cout << "Subtraction:" << std::endl;
  (t1 - t2).print();

  std::cout << "Element-wise Multiplication:" << std::endl;
  (t1 * t2).print();

  std::cout << "Element-wise Division:" << std::endl;
  (t1 / t2).print();

  std::cout << "Negation (-Tensor 1):" << std::endl;
  (-t1).print();

  std::cout << "Sum: " << t1.sum() << std::endl;
  std::cout << "Mean: " << t1.mean() << std::endl;
  std::cout << "Max: " << t1.max() << std::endl;
  std::cout << "Min: " << t1.min() << std::endl;

  return 0;
}
