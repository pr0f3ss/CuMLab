#ifndef CUMLAB_TENSOR_HPP
#define CUMLAB_TENSOR_HPP

#include <iostream>
#include <stdexcept>
#include <vector>

class Tensor {
private:
  std::vector<float> data_; // Store elements in a flat array
  std::vector<int> shape_;  // Store shape (e.g., {3, 2} for a 3x2 matrix)
  int size_;                // Total number of elements

  // Compute the flattened index for multi-dimensional access
  int compute_index(std::initializer_list<int> indices) const;

public:
  /**
   * @brief Constructs a Tensor with the given shape, initializing all elements
   * to zero.
   * @param shape The shape of the tensor.
   */
  Tensor(const std::vector<int> &shape);

  /**
   * @brief Returns the shape of the tensor.
   * @return A vector representing the shape.
   */
  std::vector<int> shape() const { return shape_; }

  /**
   * @brief Returns the total number of elements in the tensor.
   * @return The total number of elements.
   */
  int size() const { return size_; }

  float &operator()(std::initializer_list<int> indices);
  float operator()(std::initializer_list<int> indices) const;

  // Element-wise operations
  Tensor operator+(const Tensor &other) const;
  Tensor operator-(const Tensor &other) const;
  Tensor operator*(const Tensor &other) const;
  Tensor operator/(const Tensor &other) const;
  Tensor operator-() const; // Unary negation

  Tensor operator*(float scalar) const; // Scalar multiplication

  // Reduction operations
  float sum() const;
  float mean() const;
  float max() const;
  float min() const;

  void print() const;
};

#endif // CUMLAB_TENSOR_HPP
