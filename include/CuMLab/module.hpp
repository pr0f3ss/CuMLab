#ifndef CUMLAB_MODULE_HPP
#define CUMLAB_MODULE_HPP

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

template <typename T> class Tensor; ///< Forward declaration of Tensor class.

/**
 * @brief Base class for all neural network modules.
 *
 * A `Module` represents a building block of a neural network, such as a layer.
 * It supports storing parameters (e.g., weights, biases) and requires
 * subclasses to implement a `forward` function.
 *
 * @tparam T The data type used in the tensor (e.g., `float`, `int`, `uint8_t`).
 */
template <typename T> class Module {
protected:
  /**
   * @brief Stores the learnable parameters of the module.
   *
   * Each parameter is identified by a string name (e.g., `"weight"`, `"bias"`)
   * and stored as a shared pointer to a `Tensor<T>`.
   */
  std::unordered_map<std::string, std::shared_ptr<Tensor<T>>> parameters_;

public:
  /**
   * @brief Default constructor.
   */
  Module() = default;

  /**
   * @brief Virtual destructor.
   */
  virtual ~Module() = default;

  /**
   * @brief Registers a trainable parameter in the module.
   *
   * This function allows adding learnable parameters (e.g., weights, biases)
   * to the module so they can be accessed later for optimization.
   *
   * @param name The name of the parameter (e.g., `"weight"`, `"bias"`).
   * @param param A shared pointer to the tensor representing the parameter.
   */
  void register_parameter(const std::string &name,
                          std::shared_ptr<Tensor<T>> param) {
    parameters_[name] = param;
  }

  /**
   * @brief Retrieves all registered parameters.
   *
   * This function is useful for optimizers, as it provides access to all
   * trainable tensors inside the module.
   *
   * @return A vector of shared pointers to the stored parameters.
   */
  std::vector<std::shared_ptr<Tensor<T>>> parameters() {
    std::vector<std::shared_ptr<Tensor<T>>> params;
    for (auto &[name, param] : parameters_) {
      params.push_back(param);
    }
    return params;
  }

  /**
   * @brief Performs the forward pass of the module.
   *
   * This is a pure virtual function that must be overridden by derived classes
   * to define how input data is transformed.
   *
   * @param input A shared pointer to the input tensor.
   * @return A shared pointer to the output tensor.
   */
  virtual std::shared_ptr<Tensor<T>>
  forward(const std::shared_ptr<Tensor<T>> &input) override = 0;

  /**
   * @brief Calls the forward function.
   *
   * This operator allows instances of `Module` to be used like functions,
   * simplifying code when chaining modules together.
   *
   * @param input A shared pointer to the input tensor.
   * @return A shared pointer to the output tensor.
   */
  std::shared_ptr<Tensor<T>>
  operator()(const std::shared_ptr<Tensor<T>> &input) {
    return forward(input);
  }

  /**
   * @brief Saves the module parameters to a file.
   *
   * This function is a placeholder for serialization logic to store the
   * model's parameters on disk.
   *
   * @param filename The name of the file to save the parameters.
   */
  void save(const std::string &filename) {
    std::cout << "Saving " << typeid(T).name() << " model to " << filename
              << std::endl;
    // TODO: Implement default serialization
  }

  /**
   * @brief Loads the module parameters from a file.
   *
   * This function is a placeholder for deserialization logic to restore the
   * model's parameters from disk.
   *
   * @param filename The name of the file to load the parameters from.
   */
  void load(const std::string &filename) {
    std::cout << "Loading " << typeid(T).name() << " model from " << filename
              << std::endl;
    // TODO: Implement default deserialization
  }
};

#endif // CUMLAB_MODULE_HPP
