# Vineyard Yield Estimation with Spatio-Temporal CNN and Vision Transformer

## Overview

This repository contains the code and resources for implementing spatio-temporal Convolutional Neural Networks (CNN) and Vision Transformer (ViT) models to perform vineyard yield estimation, focusing on the regression of imbalanced data. Vineyard yield estimation plays a crucial role in agriculture for optimizing crop management and harvest planning.

In this project, we explore the use of state-of-the-art deep learning architectures, combining spatio-temporal CNNs and ViT models, to accurately predict vineyard yields. We address the challenge of handling imbalanced data by providing a new custom loss function and Label Density Smoothing (LDS). The results provide options for various algorithms for comparison, including Dense Weighting (DW), Lobel Density Smoothing (LDS), and Class Balancing (CB).

## Features

- Implementation of spatio-temporal CNN and Vision Transformer models for vineyard yield estimation.
- Custom data loaders for handling imbalanced data and applying data resampling techniques.
- Support for different reweighting strategies, including Dense Weighting (DW), Local Density Smoothing (LDS), and Class Balancing (CB).


## Requirements

Before running the code, ensure you have the following dependencies installed:

- Python 3.11
- PyTorch > 2.0
- NumPy
- Pandas
- Matplotlib
- Seaborn (for visualization)


```bash
python train.py --exp_name my_experiment --batch_size 64 --in_channels 4 --dropout 0.1 --ldsks 10 --ldssigma 8 --alphs 3.9 --betha 4 --lr 0.0001 --wd 0.0001 --epochs 50 --loss mse --reweight dw


