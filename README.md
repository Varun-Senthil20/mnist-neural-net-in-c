# 🧠 MNIST Neural Network in C

A basic network implementation in C for MNIST digit classification with multi-threading support.

---

## 🚀 Quick Start

### 📥 Data Setup
```bash
make data_download
```
Downloads the MNIST dataset and sets up the data folder structure.

### 🔨 Build & Compile
**Prerequisites:**
OpenBLAS library is required. Install it using:
*Note: The exact command may differ for different Linux distributions.*
```bash
sudo apt-get install libopenblas-dev
```
Compiling:
```bash
make run
```
Compiles both training and inference code, generating two executables:
- `train.o` - Training executable
- `test.o` - Testing/Inference executable
---

## 🏃‍♂️ Usage

### Training
```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=2 ./train.o -n 2
```
**Parameters:**
- `OPENBLAS_NUM_THREADS=4` - Number of threads for BLAS operations
- `OMP_NUM_THREADS=2` - Number of OpenMP threads
- `-n 2` - Number of hidden layers in the model

### Testing
```bash
./test.o
```
Runs inference on any saved model from the training code, regardless of the number of hidden layers.

---

## 🏗️ Architecture

The model supports variable hidden layers specified during training. The testing executable automatically adapts to any saved model configuration.
