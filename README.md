# Spiking Neural Network for MNIST Classification in PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

This project provides a clear and well-documented implementation of a Spiking Neural Network (SNN) for classifying MNIST handwritten digits. It is built entirely in PyTorch and serves as an excellent educational resource for understanding how to simulate SNNs, handle their temporal dynamics, and train them using surrogate gradients.

![infer figure](infer-figure.jpeg)
=======
## ✨ Features

- **Modular SNN Layers**: The `layer.py` file provides reusable building blocks like `LIFSpike` neurons and time-aware wrappers (`tdLayer`, `SeqToANNContainer`) for standard PyTorch layers.
- **MLP-based SNN Architecture**: A simple yet effective Multi-Layer Perceptron (MLP) structure is used, making the network easy to understand.
- **Surrogate Gradient Training**: Implements a custom `torch.autograd.Function` (`ZIF`) to enable backpropagation through the non-differentiable spiking process.
- **Complete Workflow**: Includes scripts for training (`train.py`) and inference (`test.py`), covering the entire machine learning pipeline.
- **Interactive Inference**: The inference script allows testing on both random MNIST samples and custom user-provided images, with clear visualizations.

---

## 📂 Project Structure

Based on the provided code, the project is organized as follows:

```
.
├── checkpoints/         # Stores trained model weights (e.g., mnist_mlp.pth)
├── data/                # Stores the MNIST dataset (auto-downloaded)
├── layer.py             # Core SNN components: LIF neuron, surrogate gradient, layer wrappers
├── model.py             # Defines the SNN network architecture (MLP)
├── train.py             # Script for model training and evaluation
├── test.py              # Script for inference and visualization
├── requirements.txt     # Python dependencies for the project
└── README.md            # This file
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/[Your-GitHub-Username]/[Your-Repo-Name].git
cd [Your-Repo-Name]
```

### 2. Set Up Environment and Install Dependencies

Using a virtual environment is highly recommended.

```bash
# Create and activate a conda environment
conda create -n snn_mnist python=3.8
conda activate snn_mnist

# Install the required packages from requirements.txt
pip install -r requirements.txt
```

### 3. Train the Model

Run the training script. It will automatically download the MNIST dataset and save the best model weights to the `checkpoints/` directory.

```bash
python train.py
```

### 4. Run Inference

Execute the inference script to see the model in action.

```bash
python test.py
```

The script will prompt you to:
- **Choose 1**: To test with a random image from the MNIST test set.
- **Choose 2**: To provide a path to your own image of a handwritten digit.🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- Special thanks to [脉冲神经网络实战课程](https://space.bilibili.com/1765043733).