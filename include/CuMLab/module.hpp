#ifndef CUMLAB_MODULE_H
#define CUMLAB_MODULE_H

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class Tensor;

class Module {
protected:
  std::unordered_map<std::string, std::shared_ptr<Tensor>> parameters_;

public:
  Module() = default;
  virtual ~Module() = default;

  // Register a parameter (e.g., weights, biases)
  void register_parameter(const std::string &name,
                          std::shared_ptr<Tensor> param) {
    parameters_[name] = param;
  }

  // Get all parameters (for optimization)
  std::vector<std::shared_ptr<Tensor>> parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto &[name, param] : parameters_) {
      params.push_back(param);
    }
    return params;
  }

  // Forward pass function (must be overridden)
  virtual std::shared_ptr<Tensor>
  forward(const std::shared_ptr<Tensor> &input) = 0;

  // Call operator
  std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor> &input) {
    return forward(input);
  }

  // Save model parameters
  void save(const std::string &filename) {
    std::cout << "Saving model to " << filename << std::endl;
    // TODO: Serialization logic
  }

  // Load model parameters
  void load(const std::string &filename) {
    std::cout << "Loading model from " << filename << std::endl;
    // TODO: Deserialization logic
  }
};

#endif // CUMLAB_MODULE_H
