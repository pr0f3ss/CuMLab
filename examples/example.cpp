#include "CuMLab/linear.hpp"
#include "CuMLab/tensor.hpp"
#include <iostream>

int main() {

  // Define a linear layer
  std::shared_ptr<Module<float>> layer =
      std::make_shared<CuMLab::Linear<float>>(4, 2);

  // Create an input tensor
  auto input = std::make_shared<Tensor<float>>(std::vector<int>{1, 4});

  // Run the forward pass
  auto output = layer->forward(input);

  std::cout << "Output Tensor:\n";
  output->print();

  return 0;
}
