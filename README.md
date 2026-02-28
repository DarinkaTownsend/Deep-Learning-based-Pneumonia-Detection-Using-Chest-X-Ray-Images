# Chest X-Ray Pneumonia Classification using ResNet-18

Author: Darinka Townsend\
Course: CAP 5516 - Medical Image Computing (Spring 2026)
Assignment #1: Deep Learning-based Pneumonia Detection Using Chest X-Ray Images

------------------------------------------------------------------------

## Project Overview

This project investigates different training strategies for binary
classification of chest X-ray images:

-   NORMAL
-   PNEUMONIA

The objective is to evaluate the impact of: 
1. Training from scratch 
2.Transfer learning (ImageNet pretraining) 
3. Class imbalance mitigation (WeightedRandomSampler)

------------------------------------------------------------------------

## Dataset

The dataset used is the publicly available Kaggle dataset: Chest X-Ray
Images (Pneumonia) by paultimothymooney.

The validation set contains only 16 images, leading to noisy
validation curves. Final evaluation is based on the test set (624
images).

------------------------------------------------------------------------

## Experimental Setup

-   Backbone: ResNet-18
-   Input size: 224x224
-   Optimizer: Adam
-   Loss: CrossEntropyLoss
-   Batch size: 32
-   Device: GPU 

------------------------------------------------------------------------

# Experiments

## Task 1.1 --- Training from Scratch

ResNet-18 initialized with random weights.

Test Results: - Accuracy: 0.7372 - Macro Precision: 0.7964 - Macro
Recall: 0.6590 - Macro F1: 0.6595

Observation: Model strongly biased toward Pneumonia class. Low recall
for NORMAL.

------------------------------------------------------------------------

## Task 1.2 --- Transfer Learning

ResNet-18 initialized with ImageNet pretrained weights.

Test Results: - Accuracy: 0.7965 - Macro Precision: 0.8731 - Macro
Recall: 0.7295 - Macro F1: 0.7447

Observation: Transfer learning significantly improves feature extraction
and overall performance. Class imbalance still affects NORMAL detection.

------------------------------------------------------------------------

## Task 1.2.1 --- Balanced Training (WeightedRandomSampler)

To address class imbalance, a WeightedRandomSampler was applied during
training.

Test Results: - Accuracy: 0.8365 - Macro Precision: 0.8873 - Macro
Recall: 0.7846 - Macro F1: 0.8047

Observation: Balanced sampling reduces model bias and significantly
improves minority class detection while maintaining high Pneumonia
recall.

------------------------------------------------------------------------

## Final Comparison

  Model        Accuracy   Macro Precision   Macro Recall   Macro F1
  ------------ ---------- ----------------- -------------- ----------
  Task 1.1     0.7372     0.7964            0.6590         0.6595
  Task 1.2     0.7965     0.8731            0.7295         0.7447
  Task 1.2.1   0.8365     0.8873            0.7846         0.8047

The final model (Task 1.2.1) achieves the best overall balance between
classes and the highest macro F1-score.

------------------------------------------------------------------------

## How to Run

1.  Install dependencies: pip install -r requirements.txt

2.  Open Medica_Image_Computing_Assignment_1.ipynb

3.  Download dataset via kagglehub

4.  Run all cells to reproduce experiments

------------------------------------------------------------------------

## Notes

-   Validation curves may appear unstable due to small validation size.
-   Test set results are used for final reporting.
