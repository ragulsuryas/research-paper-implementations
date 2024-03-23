# LeNet Implementation with PyTorch

This repository contains an implementation of the LeNet convolutional neural network using PyTorch.

## Overview

LeNet is a classic convolutional neural network architecture designed for handwritten digit recognition. It was proposed by Yann LeCun et al. in 1998 and has been widely used in various image classification tasks.

## Model Architecture

The implemented LeNet model consists of the following layers:

1. Convolutional Layer 1: Input channels -> 6 output channels, kernel size of 5x5, stride of 1x1, ReLU activation.
2. MaxPooling Layer 1: 2x2 kernel size, stride of 2x2.
3. Convolutional Layer 2: 6 input channels -> 16 output channels, kernel size of 5x5, stride of 1x1, ReLU activation.
4. MaxPooling Layer 2: 2x2 kernel size, stride of 2x2.
5. Convolutional Layer 3: 16 input channels -> 120 output channels, kernel size of 5x5, stride of 1x1, ReLU activation.
6. Fully Connected Layer 1: 120 input features -> 84 output features, ReLU activation.
7. Fully Connected Layer 2: 84 input features -> Number of classes output features.

## Usage

To use the LeNet model, follow these steps:

1. Ensure you have PyTorch installed. If not, you can install it via pip:


2. Run the `LeNet.py` file. This will initialize the LeNet model, perform a forward pass with random input data, and print the shape of the output tensor.


## Requirements

- Python 3.x
- PyTorch
- CUDA (optional, for GPU acceleration)
