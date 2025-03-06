#include "CuMLab/tensor.hpp"

namespace CuMLab {

// ─────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────
template <typename T>
Tensor<T>::Tensor(const std::vector<int> &shape) : shape_(shape) {
  size_ = 1;
  for (int dim : shape)
    size_ *= dim;
  data_.resize(size_, static_cast<T>(0)); // Initialize with zeros
}

// ─────────────────────────────────────────────────────
// Accessors
// ─────────────────────────────────────────────────────
template <typename T> std::vector<int> Tensor<T>::shape() const {
  return shape_;
}

template <typename T> int Tensor<T>::size() const { return size_; }

// ─────────────────────────────────────────────────────
// Element Access
// ─────────────────────────────────────────────────────
template <typename T>
T &Tensor<T>::operator()(std::initializer_list<int> indices) {
  int index = 0, multiplier = 1;
  auto it = indices.begin();
  for (size_t i = 0; i < shape_.size(); ++i) {
    index += (*(it + i)) * multiplier;
    multiplier *= shape_[i];
  }
  return data_[index];
}

template <typename T>
T Tensor<T>::operator()(std::initializer_list<int> indices) const {
  int index = 0, multiplier = 1;
  auto it = indices.begin();
  for (size_t i = 0; i < shape_.size(); ++i) {
    index += (*(it + i)) * multiplier;
    multiplier *= shape_[i];
  }
  return data_[index];
}

// ─────────────────────────────────────────────────────
// Element-Wise Operations
// ─────────────────────────────────────────────────────
template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T> &other) const {
  if (shape_ != other.shape_)
    throw std::invalid_argument("Shape mismatch in addition");

  Tensor<T> result(shape_);
  for (size_t i = 0; i < data_.size(); ++i) {
    result.data_[i] = data_[i] + other.data_[i];
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T> &other) const {
  if (shape_ != other.shape_)
    throw std::invalid_argument("Shape mismatch in subtraction");

  Tensor<T> result(shape_);
  for (size_t i = 0; i < data_.size(); ++i) {
    result.data_[i] = data_[i] - other.data_[i];
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T> &other) const {
  if (shape_ != other.shape_)
    throw std::invalid_argument("Shape mismatch in multiplication");

  Tensor<T> result(shape_);
  for (size_t i = 0; i < data_.size(); ++i) {
    result.data_[i] = data_[i] * other.data_[i];
  }
  return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T> &other) const {
  if (shape_ != other.shape_)
    throw std::invalid_argument("Shape mismatch in division");

  Tensor<T> result(shape_);
  for (size_t i = 0; i < data_.size(); ++i) {
    if (other.data_[i] == static_cast<T>(0))
      throw std::runtime_error("Division by zero");
    result.data_[i] = data_[i] / other.data_[i];
  }
  return result;
}

template <typename T> Tensor<T> Tensor<T>::operator-() const {
  Tensor<T> result(shape_);
  for (size_t i = 0; i < data_.size(); ++i) {
    result.data_[i] = -data_[i];
  }
  return result;
}

// ─────────────────────────────────────────────────────
// Reduction Operations
// ─────────────────────────────────────────────────────
template <typename T> T Tensor<T>::sum() const {
  T total = 0;
  for (T val : data_)
    total += val;
  return total;
}

template <typename T> T Tensor<T>::mean() const {
  if (size_ == 0)
    throw std::runtime_error("Cannot compute mean of an empty tensor.");
  return sum() / static_cast<T>(size_);
}

template <typename T> T Tensor<T>::max() const {
  if (data_.empty())
    throw std::runtime_error("Tensor is empty.");
  return *std::max_element(data_.begin(), data_.end());
}

template <typename T> T Tensor<T>::min() const {
  if (data_.empty())
    throw std::runtime_error("Tensor is empty.");
  return *std::min_element(data_.begin(), data_.end());
}

// ─────────────────────────────────────────────────────
// Debug Instructions
// ─────────────────────────────────────────────────────
template <typename T> void Tensor<T>::print() const {
  std::cout << "Shape: (";
  for (size_t i = 0; i < shape_.size(); ++i) {
    std::cout << shape_[i];
    if (i < shape_.size() - 1)
      std::cout << ", ";
  }
  std::cout << ")\n";

  // Print data in [x y z] format
  std::cout << "Values: [";
  for (size_t i = 0; i < data_.size(); ++i) {
    std::cout << data_[i];
    if (i < data_.size() - 1)
      std::cout << " ";
  }
  std::cout << "]\n";
}
// ─────────────────────────────────────────────────────
// Explicit Template Instantiations
// ─────────────────────────────────────────────────────
template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;
template class Tensor<uint8_t>;

} // namespace CuMLab
