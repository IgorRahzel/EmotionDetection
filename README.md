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

For this task, I used **transfer learning** with a pretrained **ResNet18** model from the torchvision library. The original fully connected (fc) layer of the ResNet was replaced to suit the facial expression recognition task, which has 7 output classes (angry, disgust, fear, happy, neutral, sad, surprise).

The final architecture looks like this:

```python
class FaceModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.model = models.resnet18(pretrained=True)
      self.num_ftrs = self.model.fc.in_features
      self.model.fc = nn.Linear(self.num_ftrs, 7)

  def forward(self, x):
      x = self.model(x)
      return x
```

## Facial Expression Categories
The model can predict the following facial expression classes:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

  ## Key Functions

- **`train()`**: 
  This function trains the model over several epochs, calculates the loss, and measures the accuracy for both training and validation datasets. It also includes a progress bar using `tqdm` to monitor the training progress.

  **Parameters**:
  - `model`: The neural network model to be trained.
  - `trainloader`: DataLoader for the training dataset.
  - `validloader`: DataLoader for the validation dataset.
  - `optimizer`: Optimizer for updating the model weights.
  - `num_epochs`: Number of training epochs.

  **Example**:
  ```python
  train_loss, train_acc, valid_loss, valid_acc = train(model, trainloader, validloader, optimizer, num_epochs=20)'
  ```

  - **`view_results()`**: 
  This function is used to visualize the model’s predictions compared to the actual labels. It displays images and their corresponding predicted and actual emotion labels.

  **Parameters**:
  - `images`: A batch of images.
  - `original_label`: The corresponding ground-truth labels for the images.

  **Example**:
  ```python
  view_results(images, labels)
  ```
  
  
