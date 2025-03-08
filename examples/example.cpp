#include "CuMLab/linear.hpp"
#include "CuMLab/tensor.hpp"
#include <iostream>

int main() {
  using namespace CuMLab;

  // Using float
  std::shared_ptr<Module<float>> layer_f =
      std::make_shared<Linear<float>>(4, 2);
  auto input_f = std::make_shared<Tensor<float>>(std::vector<int>{1, 4});
  auto output_f = layer_f->forward(input_f);
  std::cout << "Float Output Tensor:\n";
  output_f->print();

  // Using int
  std::shared_ptr<Module<int>> layer_i = std::make_shared<Linear<int>>(4, 2);
  auto input_i = std::make_shared<Tensor<int>>(std::vector<int>{1, 4});
  auto output_i = layer_i->forward(input_i);
  std::cout << "Int Output Tensor:\n";
  output_i->print();

  return 0;
}
