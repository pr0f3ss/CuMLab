#ifndef CUMLAB_TENSOR_HPP
#define CUMLAB_TENSOR_HPP

#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

class Tensor {
private:
  std::vector<float> data_; // Stores elements
  std::vector<int> shape_;  // Store shapes (e.g., {3, 2} for a 3x2 matrix)
  int size_;                // Total number of elements

  // Compute the flattened index for multi-dimensional access
  int compute_index(std::initializer_list<int> indices) const;

public:
  // Constructor: Create tensor with a shape, initialize all values to zero
  Tensor(std::initializer_list<int> shape);

  // Access shape
  std::vector<int> shape() const { return shape_; }

  // Access size (total elements)
  int size() const { return size_; }

  // Get & Set element-wise access
  float &operator()(std::initializer_list<int> indices);
  float operator()(std::initializer_list<int> indices) const;

  // Basic operations
  Tensor operator+(const Tensor &other) const;
  Tensor operator*(float scalar) const;

  // Print tensor contents
  void print() const;
};

#endif // CUMLAB_TENSOR_HPP
