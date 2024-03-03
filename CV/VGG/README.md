# PyTorch Implementation of VGG Architecture

This repository contains a PyTorch implementation of the VGG (Visual Geometry Group) architecture.

## Introduction

VGG is a convolutional neural network architecture proposed by the Visual Geometry Group at the University of Oxford. It is widely used for image classification tasks. The architecture consists of multiple convolutional layers followed by max-pooling layers and fully connected layers.

## Implementation

The `VGG` class in `vgg.py` implements the VGG architecture using PyTorch. It includes methods for creating the convolutional layers and fully connected layers based on the specified VGG variant (e.g., VGG11, VGG13, VGG16, VGG19).

### Usage

To use the VGG model for image classification tasks, you can follow these steps:

1. Import the `VGG` class from `vgg.py`:

    ```python
    from vgg import VGG
    ```

#### Total Parameters

The total number of parameters in the VGG model implemented in this repository varies depending on the variant used. For example, the VGG16 model has approximately 138 million parameters.

### Pre-trained Weights

Pre-trained weights for the VGG models on the ImageNet dataset are available from PyTorch's `torchvision` library. You can load pre-trained weights using the following code:

```python
import torchvision.models as models

# Load pre-trained VGG model
pretrained_vgg = models.vgg16(pretrained=True)
```

### Research Paper

The original VGG architecture was introduced in the following research paper:
[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

### License

This project is licensed under the MIT License.
