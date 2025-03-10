# Sign Language Recognition with Autoencoder and XGBoost

This project implements a sign language recognition system using a combination of an Autoencoder for dimensionality reduction and XGBoost for classification.

## Overview

The system processes the Sign Language MNIST dataset, which contains grayscale images of hand gestures representing different letters in the American Sign Language alphabet. The approach consists of two main components:

1. **Autoencoder**: Reduces the dimensionality of the input images from 784 features (28x28 pixels) to 32 features while preserving essential information.
2. **XGBoost Classifier**: Uses the encoded features to classify the hand gestures into their respective letter categories.

## Dataset

The Sign Language MNIST dataset contains:
- Training set: 27,455 examples
- Test set: 7,172 examples
- 24 classes (excluding J and Z which require motion)
- Each image is a 28x28 grayscale image (784 features)

## Implementation Details

### Preprocessing
- Standard scaling of pixel values
- SMOTE for class balancing (ensures equal representation of all classes)
- Label reindexing for contiguous class labels

### Autoencoder Architecture
- Input layer: 784 neurons (28x28 pixels)
- Encoder: 128 → 64 → 32 neurons with ReLU activation
- Decoder: 32 → 64 → 128 → 784 neurons with ReLU and sigmoid activations
- Trained with Mean Squared Error loss for 50 epochs

### XGBoost Classifier
- 200 estimators with max depth of 6
- Learning rate: 0.1
- Objective: multi:softprob for multi-class probability
- Evaluation metrics: mlogloss, merror
- Regularization parameters: gamma=0.1, alpha=0.1, lambda=1.0

## Results

The model evaluation includes:
- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Feature importance analysis

## Output Files

All results are saved in the `/Users/leopard/Desktop/NIBM/ml2/Sign-Language/Output/Autoencoder/XGBoost/` directory:

- `xgboost_model.json`: Saved XGBoost model
- `feature_importance_plots.png`: Various feature importance visualizations
- `learning_curves.png`: Training and validation metrics over iterations
- `confusion_matrix.png`: Confusion matrix visualization
- `classification_report.txt`: Detailed classification metrics

## How to Run

1. Ensure you have all required dependencies installed: