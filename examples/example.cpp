#include "CuMLab/module.hpp"
#include <iostream>

int main() {
  std::cout << "Testing CuMLab Module System..." << std::endl;

  // Dummy Linear Layer
  class Linear : public Module {
  public:
    Linear() {
      register_parameter("weight", std::make_shared<Tensor>());
      register_parameter("bias", std::make_shared<Tensor>());
    }

    std::shared_ptr<Tensor>
    forward(const std::shared_ptr<Tensor> &input) override {
      std::cout << "Forward pass in Linear Layer" << std::endl;
      return input;
    }
  };

  Linear layer;
  auto output = layer(std::make_shared<Tensor>());

  std::cout << "CuMLab compiled and ran successfully!" << std::endl;
  return 0;
}
