# Emotion Detection from Facial Expressions

This repository contains the implementation of a **Facial Expression Recognition** model using PyTorch. The model is designed to classify facial expressions from images into several categories such as anger, happiness, sadness, and more.

## Overview

The notebook provides a step-by-step approach to:
- Loading and preprocessing the dataset.
- Building a convolutional neural network (CNN) architecture using PyTorch.
- Training the model on facial expression data.
- Evaluating the model's performance.
- Visualizing model predictions and comparing them with the actual labels.

## Dataset

The dataset used for training and testing the model consists of labeled images of faces, each categorized into one of several facial expressions. Each image is processed and loaded into the model using PyTorch's `DataLoader`.

## Model Architecture

The model is based on a convolutional neural network (CNN) and includes:
- Multiple convolutional layers for feature extraction.
- Fully connected layers for classification.
- The final layer outputs a probability distribution over the possible facial expression categories.

### Facial Expression Categories
The model can predict the following facial expression classes:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise
