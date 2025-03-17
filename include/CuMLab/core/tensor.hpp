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
// Tensor class template
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
// Template inline helper function definitions
// ─────────────────────────────────────────────────────────────────────────

/**
 * @brief Computes the broadcasted shape of two tensor shapes.
 */
inline std::vector<int> broadcasted_shape(const std::vector<int> &lhs_shape,
                                          const std::vector<int> &rhs_shape) {
  // Align from right to left
  int lhs_ndim = static_cast<int>(lhs_shape.size());
  int rhs_ndim = static_cast<int>(rhs_shape.size());
  int ndim = std::max(lhs_ndim, rhs_ndim);

  std::vector<int> out_shape(ndim);

  for (int i = 0; i < ndim; ++i) {
    // Get dim from right (end). If i >= lhs_ndim, treat that as dimension=1
    int lhs_dim = (lhs_ndim - 1 - i >= 0) ? lhs_shape[lhs_ndim - 1 - i] : 1;
    int rhs_dim = (rhs_ndim - 1 - i >= 0) ? rhs_shape[rhs_ndim - 1 - i] : 1;

    if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
      throw std::invalid_argument(
          "Shapes are not broadcastable (conflicting dims).");
    }
    out_shape[ndim - 1 - i] = std::max(lhs_dim, rhs_dim);
  }
  return out_shape;
}

/**
 * @brief Converts a flat index into an N-D coordinate given a shape
 */
inline std::vector<int> unravel_index(int flat_idx,
                                      const std::vector<int> &shape) {
  std::vector<int> coords(shape.size());
  int stride = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    stride *= shape[i];
  }
  // Go from first dimension to last
  for (size_t dim = 0; dim < shape.size(); ++dim) {
    stride /= shape[dim];
    coords[dim] = flat_idx / stride;
    flat_idx = flat_idx % stride;
  }
  return coords;
}

/**
 * @brief Converts an N-D coordinate back to a flat index
 */
inline int ravel_index(const std::vector<int> &coords,
                       const std::vector<int> &shape) {
  int flat_idx = 0;
  int stride = 1;
  // Typically row-major: idx = coords[0] * (prod of shape[1..end]) +
  // coords[1]*... + ...
  for (size_t dim = 0; dim < shape.size(); ++dim) {
    flat_idx *= shape[dim];
    flat_idx += coords[dim];
  }
  return flat_idx;
}

/**
 * @brief Adjust out_coords for broadcasting if in_shape dimension is 1
 *    i.e., if a dimension is 1, the coordinate is forced to 0 in that dimension
 */
inline int broadcasted_offset(const std::vector<int> &out_coords,
                              const std::vector<int> &in_shape) {
  // e.g., if in_shape has 3 dims but out_shape has 4, treat extra as 1
  // or if in_shape[dim] == 1, force out_coords[dim] to 0
  int in_ndim = static_cast<int>(in_shape.size());
  int out_ndim = static_cast<int>(out_coords.size());

  // Build a coords set that matches in_shape's rank.
  // We align from right to left, ignoring leading dims that may not exist.
  // Or simply pad in_shape with 1s in the front for ease.
  int ndim = std::max(in_ndim, out_ndim);

  // Build a "local_coords" vector that has the same size as in_shape.
  // Then ravel it to get the final offset.
  std::vector<int> local_coords(in_ndim);

  // Start from the right
  for (int i = 0; i < ndim; ++i) {
    int out_idx = out_ndim - 1 - i; // coords in out_coords
    int in_idx = in_ndim - 1 - i;   // coords in in_shape

    int coord = 0;
    if (out_idx >= 0)
      coord = out_coords[out_idx]; //  if valid

    int shape_dim = (in_idx >= 0) ? in_shape[in_idx] : 1;

    if (shape_dim == 1) {
      // broadcast dimension, so force coords=0
      coord = 0;
    } else if (coord >= shape_dim) {
      // theoretically won't happen because we validated broadcasted shape
      throw std::runtime_error("coord out of bounds for broadcasted_offset");
    }

    if (in_idx >= 0) {
      local_coords[in_idx] = coord;
    }
  }
  return ravel_index(local_coords, in_shape);
}

// ─────────────────────────────────────────────────────────────────────────
// Template inline function definitions
// ─────────────────────────────────────────────────────────────────────────

/**
 * @brief Performs tensor addition with broadcasting.
 */
template <typename T>
std::shared_ptr<Tensor<T>> operator+(const std::shared_ptr<Tensor<T>> &lhs,
                                     const std::shared_ptr<Tensor<T>> &rhs) {
  if (!lhs || !rhs) {
    throw std::invalid_argument("Null pointer passed to operator+");
  }

  // 1) Compute the broadcasted shape
  std::vector<int> out_shape = broadcasted_shape(lhs->shape_, rhs->shape_);

  bool requires_grad = (lhs->requires_grad_ || rhs->requires_grad_);
  // 2) Allocate the result
  auto result = std::make_shared<Tensor<T>>(out_shape, requires_grad);

  // 3) Forward pass: for each index in result, figure out where to read from
  // lhs and rhs
  int out_size = result->size_;
  for (int i = 0; i < out_size; ++i) {
    // unravel 'i' in out_shape
    auto coords = unravel_index(i, out_shape);

    // figure out the offset in lhs
    int lhs_offset = broadcasted_offset(coords, lhs->shape_);
    // figure out the offset in rhs
    int rhs_offset = broadcasted_offset(coords, rhs->shape_);

    // sum
    result->data_[i] = lhs->data_[lhs_offset] + rhs->data_[rhs_offset];
  }

  // 4) If we need grad, define the backward function
  if (requires_grad) {
    result->grad_fn_ = [lhs, rhs, result, out_shape]() {
      // We'll need to do a similar loop over all elements in
      // result->grad_->data_ and accumulate into lhs->grad_ and rhs->grad_.
      const int out_size = result->size_;
      // Make sure lhs->grad_ and rhs->grad_ exist if needed
      if (lhs->requires_grad_ && !lhs->grad_) {
        lhs->grad_ = std::make_shared<Tensor<T>>(lhs->shape_, false);
      }
      if (rhs->requires_grad_ && !rhs->grad_) {
        rhs->grad_ = std::make_shared<Tensor<T>>(rhs->shape_, false);
      }

      for (int i = 0; i < out_size; ++i) {
        T grad_val = result->grad_->data_[i];
        auto coords = unravel_index(i, out_shape);

        if (lhs->requires_grad_) {
          int lhs_offset = broadcasted_offset(coords, lhs->shape_);
          lhs->grad_->data_[lhs_offset] += grad_val;
        }
        if (rhs->requires_grad_) {
          int rhs_offset = broadcasted_offset(coords, rhs->shape_);
          rhs->grad_->data_[rhs_offset] += grad_val;
        }
      }

      if (lhs->grad_fn_) {
        lhs->grad_fn_();
      }

      if (rhs->grad_fn_) {
        rhs->grad_fn_();
      }
    };
  }

  return result;
}

/**
 * @brief Performs natrix multiplication of two tensors.
 */
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

// Explicit template instantiations
extern template class Tensor<int>;
extern template class Tensor<float>;
extern template class Tensor<double>;
extern template class Tensor<uint8_t>;

} // namespace CuMLab

#endif // CUMLAB_TENSOR_HPP
