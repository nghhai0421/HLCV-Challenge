# HLCV Fruit Image Classification Challenge

This repository documents my experiments for the University of Mannheim's Higher Level Computer Vision challenge: classifying images into 10 fruit categories with a limit of **10,000 parameters** across the complete inference pipeline. The provided dataset contains **2,300 training images** and **1,025 validation images**. Final performance is evaluated on a hidden test set on a remote server.

I use this challenge to explore the evolution of convolutional neural networks (CNNs), learning how the components found in modern architectures work and how to adapt them to a small parameter budget.

## Exploring CNN architectures

The experiments progress from simple CNN baselines and larger convolution kernels to ResNet-, ConvNeXt V2-, and UniRepLKNet-inspired models. Along the way, I explore:

- Residual connections for learning refinements to existing features.
- Depthwise convolutions and pointwise channel mixing for parameter efficiency.
- Inverted bottlenecks that expand channels before projecting them back down.
- Spatial attention to emphasize informative image regions, and squeeze-and-excitation (SE) attention to reweight channels.
- Global Response Normalization (GRN), used in ConvNeXt V2 to encourage competition between channel responses.
- Larger receptive fields through larger kernels and dilated convolutions.

There is also a side experiment with Neural Cellular Automata (NCA), exploring iterative feature updates as an alternative to conventional CNN layers.

## Training and optimization experiments

Alongside architecture changes, I explore AdamW, learning-rate warmup and cosine annealing, exponential moving averages (EMA) of model weights, label smoothing, gradient clipping, and data augmentation.

The focus is on understanding how these choices affect accuracy, generalization, and the trade-offs imposed by the parameter limit. The `models_archive/` folder records the model progression, with saved implementations, run notes, configurations, logs, checkpoints, and evaluation reports where available.

## Current best attempt

The best validation result so far is **79.02%** with **BudgetUniRepLKNet**, a **9,986-parameter** model combining residual connections, depthwise convolutions, SE channel attention, GRN, and spatial attention. The saved remote evaluation reports **48.30% accuracy on the hidden test set**.
