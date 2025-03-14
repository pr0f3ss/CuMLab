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

  if (requires_grad_)
    grad_ = std::make_shared<Tensor<T>>(shape_);
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

  if (!grad_)
    grad_ = std::make_shared<Tensor<T>>(shape_);

  for (size_t i = 0; i < grad_->size(); ++i) {
    (*grad_)({static_cast<int>(i)}) = static_cast<T>(1);
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
template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T> &other) const {
  // Case 1: Shapes match exactly (element-wise addition)
  if (shape_ == other.shape_) {
    Tensor<T> result(shape_, requires_grad_ || other.requires_grad_);
    for (size_t i = 0; i < data_.size(); ++i) {
      result.data_[i] = data_[i] + other.data_[i];
    }

    if (result.requires_grad_) {
      auto self_grad = this->grad_;
      auto other_grad = other.grad_;
      result.set_grad_fn([self_grad, other_grad]() {
        for (size_t i = 0; i < self_grad->size(); ++i) {
          (*self_grad)({static_cast<int>(i)}) += 1;
          (*other_grad)({static_cast<int>(i)}) += 1;
        }
      });
    }

    return result;
  }

  // Case 2: **Bias Broadcasting Case**
  if (other.shape_.size() == 1 && shape_.back() == other.shape_[0]) {
    Tensor<T> result(shape_, requires_grad_ || other.requires_grad_);
    for (size_t i = 0; i < static_cast<size_t>(shape_[0]); ++i) {
      for (size_t j = 0; j < static_cast<size_t>(shape_.back()); ++j) {
        result({static_cast<int>(i), static_cast<int>(j)}) =
            (*this)({static_cast<int>(i), static_cast<int>(j)}) +
            other({static_cast<int>(j)});
      }
    }

    if (result.requires_grad_) {
      auto self_grad = this->grad_;
      auto other_grad = other.grad_;

      result.set_grad_fn([self_grad, other_grad]() {
        for (size_t i = 0; i < static_cast<size_t>(self_grad->shape_[0]); ++i) {
          for (size_t j = 0; j < static_cast<size_t>(self_grad->shape_[1]);
               ++j) {
            (*self_grad)({static_cast<int>(i), static_cast<int>(j)}) += 1;
          }
        }

        for (size_t j = 0; j < static_cast<size_t>(other_grad->shape_[0]);
             ++j) {
          (*other_grad)({static_cast<int>(j)}) += 1; // Accumulate across rows
        }
      });
    }

    return result;
  }

  throw std::invalid_argument("Shape mismatch in addition: Cannot broadcast");
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T> &other) const {
  // Exact shape match (element-wise subtraction)
  if (shape_ == other.shape_) {
    Tensor<T> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
      result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
  }

  // Case 2: Broadcasting support (1D tensor -> last dimension of this tensor)
  if (other.shape_.size() == 1 && shape_.back() == other.shape_[0]) {
    Tensor<T> result(shape_);
    for (size_t i = 0; i < static_cast<size_t>(shape_[0]); ++i) { // Cast to int
      for (size_t j = 0; j < static_cast<size_t>(shape_.back());
           ++j) { // Cast to int
        result({static_cast<int>(i), static_cast<int>(j)}) =
            (*this)({static_cast<int>(i), static_cast<int>(j)}) -
            other({static_cast<int>(j)});
      }
    }
    return result;
  }

  throw std::invalid_argument(
      "Shape mismatch in subtraction: Cannot broadcast");
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T> &other) const {
  if (shape_.size() != 2 || other.shape_.size() != 2) {
    throw std::invalid_argument("Matrix multiplication requires 2D tensors.");
  }

  int rows = shape_[0];
  int colsA = shape_[1];
  int colsB = other.shape_[1];

  if (colsA != other.shape_[0]) {
    throw std::invalid_argument(
        "Shape mismatch in multiplication: Inner dimensions must match.");
  }

  Tensor<T> result(std::vector<int>{rows, colsB},
                   requires_grad_ || other.requires_grad_);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < colsB; ++j) {
      T sum = 0;
      for (int k = 0; k < colsA; ++k) {
        sum += (*this)({i, k}) * other({k, j});
      }
      result({i, j}) = sum;
    }
  }

  if (result.requires_grad_) {
    auto self_grad = this->grad_;
    auto other_grad = other.grad_;
    auto this_ptr = this;
    auto other_ptr = &other;

    result.set_grad_fn([self_grad, other_grad, this_ptr, other_ptr]() {
      for (int i = 0; i < this_ptr->shape()[0]; ++i) {
        for (int j = 0; j < other_ptr->shape()[1]; ++j) {
          for (int k = 0; k < this_ptr->shape()[1]; ++k) {
            (*self_grad)({i, k}) += (*other_ptr)({k, j}); // dZ/dA = B
            (*other_grad)({k, j}) += (*this_ptr)({i, k}); // dZ/dB = A
          }
        }
      }
    });
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
