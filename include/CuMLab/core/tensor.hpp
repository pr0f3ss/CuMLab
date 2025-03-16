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
// Forward declaration of class template
template <typename T> class Tensor;

// ─────────────────────────────────────────────────────────────────────────
// Function templates that use shared_ptr
// ─────────────────────────────────────────────────────────────────────────
template <typename U>
std::shared_ptr<Tensor<U>> operator+(const std::shared_ptr<Tensor<U>> &lhs,
                                     const std::shared_ptr<Tensor<U>> &rhs);

template <typename T>
std::shared_ptr<Tensor<T>> matmul(const std::shared_ptr<Tensor<T>> &lhs,
                                  const std::shared_ptr<Tensor<T>> &rhs);

// ─────────────────────────────────────────────────────────────────────────
// 2) Tensor class template
// ─────────────────────────────────────────────────────────────────────────
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
  // Tensor<T> operator+(const Tensor<T> &other) const;

  /**
   * @brief Performs element-wise subtraction with another tensor.
   */
  Tensor<T> operator-(const Tensor<T> &other) const;

  /**
   * @brief Performs element-wise multiplication (Hadamard product) with another
   * tensor.
   */
  // Tensor<T> operator*(const Tensor<T> &other) const;

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

  // Make operator+ / operator* overloads that accept shared_ptr friend
  // functions so they can access private data_ directly if needed.
  template <typename U>
  friend std::shared_ptr<Tensor<U>>
  operator+(const std::shared_ptr<Tensor<U>> &lhs,
            const std::shared_ptr<Tensor<U>> &rhs);

  template <typename U>
  friend std::shared_ptr<Tensor<U>>
  matmul(const std::shared_ptr<Tensor<U>> &lhs,
         const std::shared_ptr<Tensor<U>> &rhs);
};

// ─────────────────────────────────────────────────────────────────────────
// Template inline function definitions
// ─────────────────────────────────────────────────────────────────────────

// operator+ definition
template <typename U>
std::shared_ptr<Tensor<U>> operator+(const std::shared_ptr<Tensor<U>> &lhs,
                                     const std::shared_ptr<Tensor<U>> &rhs) {
  if (!lhs || !rhs) {
    throw std::invalid_argument("Null pointer passed to operator+");
  }
  // Example: check shapes match
  if (lhs->shape_ != rhs->shape_) {
    throw std::invalid_argument(
        "Shapes do not match for element-wise addition");
  }

  bool requires_grad = lhs->requires_grad_ || rhs->requires_grad_;
  auto result = std::make_shared<Tensor<U>>(lhs->shape_, requires_grad);

  // Forward pass: element-wise add
  for (int i = 0; i < lhs->size_; i++) {
    result->data_[i] = lhs->data_[i] + rhs->data_[i];
  }

  // If we need grad, define grad_fn
  if (requires_grad) {
    result->grad_fn_ = [lhs, rhs, result]() {
      // accumulate gradients to lhs
      if (lhs->requires_grad_) {
        if (!lhs->grad_) {
          lhs->grad_ = std::make_shared<Tensor<U>>(lhs->shape_, false);
        }
        for (int i = 0; i < lhs->size_; i++) {
          lhs->grad_->data_[i] += result->grad_->data_[i];
        }
      }
      // accumulate gradients to rhs
      if (rhs->requires_grad_) {
        if (!rhs->grad_) {
          rhs->grad_ = std::make_shared<Tensor<U>>(rhs->shape_, false);
        }
        for (int i = 0; i < rhs->size_; i++) {
          rhs->grad_->data_[i] += result->grad_->data_[i];
        }
      }
    };
  }

  return result;
}

template <typename T>
std::shared_ptr<Tensor<T>> matmul(const std::shared_ptr<Tensor<T>> &lhs,
                                  const std::shared_ptr<Tensor<T>> &rhs) {
  using TensorPtr = std::shared_ptr<Tensor<T>>;
  // Check that we actually have 2D shapes:
  if (lhs->shape_.size() != 2 || rhs->shape_.size() != 2) {
    throw std::invalid_argument(
        "matmul only supports 2D Tensors in this example");
  }
  int M = lhs->shape_[0];
  int K = lhs->shape_[1];
  int K2 = rhs->shape_[0];
  int N = rhs->shape_[1];

  // Check inner dims match: (K == K2)
  if (K != K2) {
    throw std::invalid_argument(
        "Inner dimensions do not match for matmul (lhs NxK, rhs KxM).");
  }

  bool requires_grad = lhs->requires_grad_ || rhs->requires_grad_;

  // Create the output shape [M, N]
  std::vector<int> out_shape = {M, N};
  auto result = std::make_shared<Tensor<T>>(out_shape, requires_grad);

  // Forward pass: standard matrix multiply
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      T sum_val = static_cast<T>(0);
      for (int k = 0; k < K; ++k) {
        sum_val += lhs->data_[i * K + k] * rhs->data_[k * N + j];
      }
      result->data_[i * N + j] = sum_val;
    }
  }

  // If grad is needed, define the backward function
  if (requires_grad) {
    result->grad_fn_ = [lhs, rhs, result, M, K, N]() {
      // dL/d(lhs) = (dL/d(result)) * rhs^T
      // dL/d(rhs) = lhs^T * (dL/d(result))

      // We'll call dL/d(result) = result->grad_->data_.
      // shape: (M,N)

      // Propagate grads to lhs
      if (lhs->requires_grad_) {
        if (!lhs->grad_) {
          lhs->grad_ = std::make_shared<Tensor<T>>(lhs->shape_, false);
        }
        // lhs->grad_ shape is (M,K)
        // For each [m,k], gradient = sum_{n} (dL/dZ[m,n] * rhs[k,n])
        for (int m = 0; m < M; ++m) {
          for (int k = 0; k < K; ++k) {
            T grad_val = 0;
            for (int n = 0; n < N; ++n) {
              grad_val +=
                  result->grad_->data_[m * N + n] * rhs->data_[k * N + n];
            }
            lhs->grad_->data_[m * K + k] += grad_val;
          }
        }
      }

      // Propagate grads to rhs
      if (rhs->requires_grad_) {
        if (!rhs->grad_) {
          rhs->grad_ = std::make_shared<Tensor<T>>(rhs->shape_, false);
        }
        // rhs->grad_ shape is (K,N)
        // For each [k,n], gradient = sum_{m} (lhs[m,k] * dL/dZ[m,n])
        for (int k = 0; k < K; ++k) {
          for (int n = 0; n < N; ++n) {
            T grad_val = 0;
            for (int m = 0; m < M; ++m) {
              grad_val +=
                  lhs->data_[m * K + k] * result->grad_->data_[m * N + n];
            }
            rhs->grad_->data_[k * N + n] += grad_val;
          }
        }
      }
    };
  }

  return result;
}

// Explicit template instantiations (optional)
extern template class Tensor<int>;
extern template class Tensor<float>;
extern template class Tensor<double>;
extern template class Tensor<uint8_t>;

} // namespace CuMLab

#endif // CUMLAB_TENSOR_HPP
