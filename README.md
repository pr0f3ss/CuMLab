# CuMLab

CuMLab is a C++ library for experimenting with low-level ML components, offering a **Tensor** data structure with autograd, modules (like `Linear` layers), and GPU acceleration (in future). It’s designed as a **learning playground** for building scalable machine-learning pipelines in C++ (and eventually CUDA).

---

## Features

- **Tensors**  
  - Multi-dimensional arrays with broadcasting support  
  - Autograd integration for backprop  
  - Element-wise, matrix ops

- **Modules**  
  - Base class for layers (`Module<T>`)  
  - Example: `Linear<T>` layer for fully-connected ops  

- **Activations & Loss**  
  - ReLU, Sigmoid, Tanh (planned)  
  - MSELoss, MAELoss, etc. (planned)

- **Optimizer** (WIP)  
  - SGD, Adam (planned)

- **Documentation**  
  - [Online Documentation](https://pr0f3ss.github.io/CuMLab/) using Doxygen + Sphinx

---

## Building & Installing

1. **Clone the Repo**
   ```bash
   git clone https://github.com/pr0f3ss/CuMLab.git
   cd CuMLab
   ```

2. **Configure & Build**
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

3. **Run Examples**
   ```bash
   ./examples/tensor_example
   ./examples/activation_example
   ./examples/example
   ```
   See the `examples/` directory for usage demos.

---

## Documentation

Full docs are generated with **Doxygen** & **Sphinx**:
1. `doxygen Doxyfile`
2. `cd docs && make html`
3. Open `docs/build/html/index.html`

Alternatively, browse the **[Online Docs](https://pr0f3ss.github.io/CuMLab/)** directly.

---

## Roadmap

- **Advanced Ops & Layers**
  - Convolution, Pooling (for CNNs)
  - RNN / LSTM (sequence tasks)
- **Loss Functions**
  - CrossEntropy, custom objectives
- **Optimizers**
  - Adam, RMSProp
- **CUDA Acceleration**
  - GPU memory, cuBLAS, cuDNN

---

## Contributing

Contributions are welcome!  
- **Open an issue** to discuss feature requests, questions, or bugs.  
- **Submit a PR** with improvements for Tensors, modules, or docs.

---

## License

[MIT License](LICENSE)

---