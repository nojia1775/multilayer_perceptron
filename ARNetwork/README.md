# 🧠 ARNetwork

**ARNetwork** is a custom C++ library designed to facilitate the development and experimentation with neural networks. It provides essential components for building and training neural networks from scratch, focusing on simplicity and educational value.

---

## 🧩 Overview

ARNetwork is a lightweight, self-contained library aimed at providing the fundamental building blocks for neural network construction and training. It is designed for educational purposes and to offer a clear understanding of how neural networks operate under the hood.

---

## ⚙️ Features

- 🧮 Custom linear algebra operations (vectors, matrices, dot products, etc.)
- 🧠 Modular neural network components (layers, activations, loss functions)
- 🔄 Forward and backward propagation
- 🛠️ Simple training loop with gradient descent
- 📈 Visualization support (via external scripts)
- 🧱 Minimal dependencies — primarily self-contained

---

## 🏗️ Architecture

### 🔢 Linear Algebra Module

This module provides essential mathematical operations:

- Vectors and matrices
- Arithmetic operations (addition, subtraction, multiplication)
- Dot products, transposition, normalization
- Utility functions for indexing and slicing

### 🧬 Neural Network Module

This module offers components to build and train neural networks:

- Layers (Dense, Activation functions)
- Forward and backward propagation
- Loss functions (e.g., Mean Squared Error)
- Weight initialization and updates

While this project mainly uses linear regression, the neural network library allows future extensions — for example, comparing regression models to small neural nets.

---


## 🖥️ Usage

1. Include the necessary headers from the `linear_algebra` and `neural_network` directories.
2. Create instances of layers and define the network architecture.
3. Implement the training loop with forward and backward passes.
4. Utilize the provided loss functions and optimization techniques.

Check out my [Ft_linear_regression](https://github.com/nojia1775/ft_linear_regression)

---
## 📜 License

This project is open-source under the **MIT License**.

```
MIT License

Copyright (c) 2025 ...

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction...

