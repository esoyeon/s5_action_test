# MNIST Model CI/CD Pipeline

![Build Status](https://github.com/{username}/{repository}/workflows/MNIST%20Model%20CI/badge.svg)

This project implements a Convolutional Neural Network (CNN) for the MNIST dataset with automated testing and continuous integration.

## Model Architecture

- Input Layer: 28x28 grayscale images
- First Convolutional Layer: 8 filters, 3x3 kernel
- MaxPooling Layer: 2x2
- Second Convolutional Layer: 16 filters, 3x3 kernel
- MaxPooling Layer: 2x2
- Flatten Layer
- Output Layer: Dense layer with 10 units (softmax activation)

## Requirements

- Python 3.8+
- TensorFlow 2.x
- pytest
- numpy

## Setup

1. Clone the repository: 