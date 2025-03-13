#ifndef CUMLAB_TENSOR_HPP
#define CUMLAB_TENSOR_HPP

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace CuMLab {

/**
 * @brief A multi-dimensional tensor class supporting generic data types.
 *
 * @tparam T The data type of the tensor elements (e.g., `float`, `int`,
 * `uint8_t`).
 */
template <typename T> class Tensor {
private:
  std::vector<T> data_;             ///< Flat storage for tensor values.
  std::vector<int> shape_;          ///< Shape of the tensor.
  int size_;                        ///< Total number of elements.
  bool requires_grad_;              ///< Flag for gradient tracking.
  std::shared_ptr<Tensor<T>> grad_; ///< Gradient storage.
  std::function<void()> grad_fn_;   ///< Function to compute gradients.

public:
  /**
   * @brief Constructs a Tensor with the given shape, initializing all elements
   * to zero.
   * @param shape The shape of the tensor.
   */
  explicit Tensor(const std::vector<int> &shape, bool requires_grad = false);

  /**
   * @brief Returns the shape of the tensor.
   * @return A vector representing the shape.
   */
  std::vector<int> shape() const;

  /**
   * @brief Returns the total number of elements in the tensor.
   * @return The total number of elements.
   */
  int size() const;

  /**
   * @brief Access or modify an element in the tensor.
   * @param indices The indices of the element to access.
   * @return A reference to the requested tensor element.
   * @throws std::out_of_range If indices are out of bounds.
   */
  T &operator()(std::initializer_list<int> indices);

  /**
   * @brief Access an element in the tensor (read-only).
   * @param indices The indices of the element to retrieve.
   * @return The value of the requested tensor element.
   * @throws std::out_of_range If indices are out of bounds.
   */
  T operator()(std::initializer_list<int> indices) const;

  // ─────────────────────────────────────────────────────
  // Gradient
  // ─────────────────────────────────────────────────────

  /**
   * @brief Returns the gradient tensor.
   */
  std::shared_ptr<Tensor<T>> grad();

  /**
   * @brief Sets the function that computes gradients for this tensor.
   */
  void set_grad_fn(std::function<void()> grad_fn);

  /**
   * @brief Performs backpropagation.
   */
  void backward();

  // ─────────────────────────────────────────────────────
  // Element-wise Operations
  // ─────────────────────────────────────────────────────

  /**
   * @brief Performs element-wise addition with another tensor.
   * @param other The tensor to add.
   * @return A new tensor containing the sum of corresponding elements.
   * @throws std::invalid_argument If tensor shapes do not match.
   */
  Tensor<T> operator+(const Tensor<T> &other) const;

  /**
   * @brief Performs element-wise subtraction with another tensor.
   */
  Tensor<T> operator-(const Tensor<T> &other) const;

  /**
   * @brief Performs element-wise multiplication (Hadamard product) with another
   * tensor.
   */
  Tensor<T> operator*(const Tensor<T> &other) const;

  /**
   * @brief Performs element-wise division with another tensor.
   */
  Tensor<T> operator/(const Tensor<T> &other) const;

  /**
   * @brief Returns a tensor with all elements negated (unary `-`).
   */
  Tensor<T> operator-() const;

  /**
   * @brief Multiplies each element of the tensor by a scalar value.
   */
  Tensor<T> operator*(T scalar) const;

  // ─────────────────────────────────────────────────────
  // Reduction Operations
  // ─────────────────────────────────────────────────────

  /**
   * @brief Computes the sum of all elements in the tensor.
   */
  T sum() const;

  /**
   * @brief Computes the mean (average) of all elements in the tensor.
   */
  T mean() const;

  /**
   * @brief Finds the maximum value in the tensor.
   */
  T max() const;

  /**
   * @brief Finds the minimum value in the tensor.
   */
  T min() const;

  /**
   * @brief Prints the tensor's shape and elements to the console.
   */
  void print() const;
};

// Explicit template instantiations (optional)
extern template class Tensor<int>;
extern template class Tensor<float>;
extern template class Tensor<double>;
extern template class Tensor<uint8_t>;

} // namespace CuMLab

#endif // CUMLAB_TENSOR_HPP
