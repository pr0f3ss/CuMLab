#include "CuMLab/tensor.hpp"

// Constructor: Initializes tensor with the given shape, all values set to zero
Tensor::Tensor(const std::vector<int> &shape) : shape_(shape) {
  size_ = 1;
  for (int dim : shape_) {
    size_ *= dim;
  }
  data_.resize(size_, 0.0f); // Initialize with zeros
}

// Compute the flattened index from multi-dimensional indices
int Tensor::compute_index(std::initializer_list<int> indices) const {
  if (indices.size() != shape_.size()) {
    throw std::out_of_range("Tensor index out of range.");
  }

  int index = 0;
  int stride = 1;
  auto shape_it = shape_.rbegin();
  auto idx_it = indices.end() - 1;

  for (; shape_it != shape_.rend(); ++shape_it, --idx_it) {
    if (*idx_it >= *shape_it || *idx_it < 0) {
      throw std::out_of_range("Tensor index out of bounds.");
    }
    index += (*idx_it) * stride;
    stride *= (*shape_it);
  }
  return index;
}

// Access (read/write) an element in the tensor
float &Tensor::operator()(std::initializer_list<int> indices) {
  return data_[compute_index(indices)];
}

// Access (read-only) an element in the tensor
float Tensor::operator()(std::initializer_list<int> indices) const {
  return data_[compute_index(indices)];
}

// Element-wise addition
Tensor Tensor::operator+(const Tensor &other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("Tensor shapes must match for addition.");
  }

  Tensor result(shape_);
  for (int i = 0; i < size_; i++) {
    result.data_[i] = data_[i] + other.data_[i];
  }
  return result;
}

// Element-wise subtraction
Tensor Tensor::operator-(const Tensor &other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("Tensor shapes must match for subtraction.");
  }

  Tensor result(shape_);
  for (int i = 0; i < size_; i++) {
    result.data_[i] = data_[i] - other.data_[i];
  }
  return result;
}

// Element-wise multiplication (Hadamard product)
Tensor Tensor::operator*(const Tensor &other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("Tensor shapes must match for multiplication.");
  }

  Tensor result(shape_);
  for (int i = 0; i < size_; i++) {
    result.data_[i] = data_[i] * other.data_[i];
  }
  return result;
}

// Element-wise division
Tensor Tensor::operator/(const Tensor &other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("Tensor shapes must match for division.");
  }

  Tensor result(shape_);
  for (int i = 0; i < size_; i++) {
    if (other.data_[i] == 0) {
      throw std::runtime_error("Division by zero in tensor operation.");
    }
    result.data_[i] = data_[i] / other.data_[i];
  }
  return result;
}

// Negation (Unary `-tensor`)
Tensor Tensor::operator-() const {
  Tensor result(shape_);
  for (int i = 0; i < size_; i++) {
    result.data_[i] = -data_[i];
  }
  return result;
}

// Scalar multiplication
Tensor Tensor::operator*(float scalar) const {
  Tensor result(shape_);
  for (int i = 0; i < size_; i++) {
    result.data_[i] = data_[i] * scalar;
  }
  return result;
}

// Sum accumulator
float Tensor::sum() const {
  float total = 0.0f;
  for (float val : data_) {
    total += val;
  }
  return total;
}

// Mean
float Tensor::mean() const {
  if (size_ == 0)
    throw std::runtime_error("Cannot compute mean of an empty tensor.");
  return sum() / size_;
}

// Max
float Tensor::max() const {
  if (data_.empty())
    throw std::runtime_error("Tensor is empty.");
  float max_val = data_[0];
  for (float val : data_) {
    if (val > max_val)
      max_val = val;
  }
  return max_val;
}

// Min
float Tensor::min() const {
  if (data_.empty())
    throw std::runtime_error("Tensor is empty.");
  float min_val = data_[0];
  for (float val : data_) {
    if (val < min_val)
      min_val = val;
  }
  return min_val;
}

// Print tensor for debugging
void Tensor::print() const {
  std::cout << "Tensor(shape=[";
  for (size_t i = 0; i < shape_.size(); i++) {
    std::cout << shape_[i] << (i < shape_.size() - 1 ? ", " : "");
  }
  std::cout << "], data=[";
  for (size_t i = 0; i < size_; i++) {
    std::cout << data_[i] << (i < size_ - 1 ? ", " : "");
  }
  std::cout << "])\n";
}
