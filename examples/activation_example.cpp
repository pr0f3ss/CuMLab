#include "CuMLab/core/tensor.hpp"
#include "CuMLab/layers/linear.hpp"
#include "CuMLab/layers/relu.hpp"
#include "CuMLab/layers/sigmoid.hpp"
#include "CuMLab/layers/tanh.hpp"
#include <iostream>


int main() {
  using namespace CuMLab;

  // Create a Linear Layer
  std::shared_ptr<Module<float>> layer = std::make_shared<Linear<float>>(4, 2);
  auto input = std::make_shared<Tensor<float>>(std::vector<int>{1, 4});

  // Initialize input tensor with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0,
                                             1.0); // Values between -1 and 1

  for (int i = 0; i < input->size(); ++i) {
    (*input)({i}) = dist(gen);
  }

  std::cout << "Initialized Input Tensor:\n";
  input->print();

  // Pass input through Linear layer
  auto output = layer->forward(input);
  std::cout << "Linear Output:\n";
  output->print();

  // Apply ReLU Activation
  std::shared_ptr<Module<float>> relu = std::make_shared<ReLU<float>>();
  auto relu_out = relu->forward(output);
  std::cout << "ReLU Output:\n";
  relu_out->print();

  // Apply Sigmoid Activation
  std::shared_ptr<Module<float>> sigmoid = std::make_shared<Sigmoid<float>>();
  auto sigmoid_out = sigmoid->forward(output);
  std::cout << "Sigmoid Output:\n";
  sigmoid_out->print();

  // Apply Tanh Activation
  std::shared_ptr<Module<float>> tanh = std::make_shared<Tanh<float>>();
  auto tanh_out = tanh->forward(output);
  std::cout << "Tanh Output:\n";
  tanh_out->print();

  return 0;
}
