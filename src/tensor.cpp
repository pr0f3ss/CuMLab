#include "CuMLab/core/tensor.hpp"

namespace CuMLab {

// ─────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────
template <typename T>
Tensor<T>::Tensor(const std::vector<int> &shape, bool requires_grad)
    : shape_(shape), requires_grad_(requires_grad) {
  size_ = 1;
  for (int dim : shape) {
    if (dim <= 0) {
      throw std::invalid_argument("Invalid tensor dimension (must be > 0).");
    }
    size_ *= dim;
  }
  data_.resize(size_, static_cast<T>(0));
}

// ─────────────────────────────────────────────────────
// Accessors
// ─────────────────────────────────────────────────────
template <typename T> std::vector<int> Tensor<T>::shape() const {
  return shape_;
}

template <typename T> int Tensor<T>::size() const { return size_; }

// ─────────────────────────────────────────────────────
// Gradient
// ─────────────────────────────────────────────────────

template <typename T> std::shared_ptr<Tensor<T>> Tensor<T>::grad() {
  return grad_;
}

template <typename T>
void Tensor<T>::set_grad_fn(std::function<void()> grad_fn) {
  grad_fn_ = std::move(grad_fn);
}

template <typename T> void Tensor<T>::backward() {
  if (!requires_grad_)
    throw std::runtime_error("Tensor does not require gradients.");

  if (!grad_) {
    grad_ = std::make_shared<Tensor<T>>(shape_);
    std::fill(grad_->data_.begin(), grad_->data_.end(), static_cast<T>(1));
  }

  if (grad_fn_)
    grad_fn_(); // Call the gradient function
}

// ─────────────────────────────────────────────────────
// Element Access
// ─────────────────────────────────────────────────────
template <typename T>
T &Tensor<T>::operator()(std::initializer_list<int> indices) {
  if (indices.size() != shape_.size() && indices.size() != 1) {
    throw std::invalid_argument(
        "Incorrect number of indices for tensor access.");
  }

  int index = 0;
  if (indices.size() == 1) {
    index = *(indices.begin()); // Direct flat indexing
    if (index < 0 || index >= size_) {
      throw std::out_of_range("Flat index out of bounds");
    }
  } else {
    int multiplier = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
      int idx = *(indices.begin() + i);
      if (idx < 0 || idx >= shape_[i]) {
        throw std::out_of_range("Multi-dimensional index out of bounds");
      }
      index += idx * multiplier;
      multiplier *= shape_[i];
    }
  }

  return data_[index];
}

template <typename T>
T Tensor<T>::operator()(std::initializer_list<int> indices) const {
  if (indices.size() != shape_.size() && indices.size() != 1) {
    throw std::invalid_argument(
        "Incorrect number of indices for tensor access.");
  }

  int index = 0;
  if (indices.size() == 1) {
    index = *(indices.begin()); // Direct flat indexing
    if (index < 0 || index >= size_) {
      throw std::out_of_range("Flat index out of bounds");
    }
  } else {
    int multiplier = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
      int idx = *(indices.begin() + i);
      if (idx < 0 || idx >= shape_[i]) {
        throw std::out_of_range("Multi-dimensional index out of bounds");
      }
      index += idx * multiplier;
      multiplier *= shape_[i];
    }
  }

  return data_[index];
}

// ─────────────────────────────────────────────────────
// Element-Wise Operations
// ─────────────────────────────────────────────────────

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
