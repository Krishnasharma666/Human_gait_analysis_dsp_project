# Human Gait Analysis & Classification using Marker Kinematics

A comprehensive machine learning pipeline for preprocessing, feature extraction, and classification of human gait biomechanics using 3D motion-capture marker data.

## 📌 Project Overview

This project analyzes 3D kinematic signals derived from optical motion capture markers placed on lower limb landmarks (`L_FCC`, `L_FM1`, `L_FM2`, `L_FM5` and corresponding right-foot markers across X, Y, and Z axes). The goal is to detect subtle variations in walking patterns and classify **Normal-like Gait** versus **Abnormal-like Gait** using Digital Signal Processing (DSP) and supervised machine learning models.

## 🛠️ Data Pipeline & Architecture

```
Raw 3D Kinematics (24 Marker Channels)
               │
               ▼
   Signal Preprocessing (DSP)
 ┌──────────────────────────────────────────────┐
 │ • Linear Detrending                          │
 │ • 4th-Order Butterworth Low-pass Filter (5Hz)│
 └──────────────────────────────────────────────┘
               │
               ▼
   Sliding Window Segmentation
 ┌──────────────────────────────────────────────┐
 │ • Window Size : 200 frames (2.0s @ 100Hz)    │
 │ • Overlap     : 50% (100-frame step)         │
 └──────────────────────────────────────────────┘
               │
               ▼
    Feature Extraction (288 Total Features)
 ┌──────────────────────────────────────────────┐
 │ • Time-Domain  : Mean, Std, Variance, RMS,   │
 │                  Min, Max, Range, Energy,    │
 │                  Skewness, Kurtosis          │
 │ • Frequency    : Dominant Frequency,         │
 │                  Spectral Centroid           │
 └──────────────────────────────────────────────┘
               │
               ▼
  Classification & Evaluation Pipeline
 ┌──────────────────────────────────────────────┐
 │ • StandardScaler Standardization             │
 │ • Models: KNN, Logistic Regression, GNB      │
 └──────────────────────────────────────────────┘

```

## ⚙️ Experimental Configuration

| 

| **Parameter** | **Value** | **Description** | 
| **Sampling Rate (**$F_s$**)** | `100.0 Hz` | Motion capture system collection frequency | 
| **Filter Cutoff** | `5.0 Hz` | Low-pass filter threshold to remove high-frequency noise | 
| **Filter Order** | `4` | Zero-phase forward-backward Butterworth filter (`sosfiltfilt`) | 
| **Window Size** | `200 samples` | 2-second temporal window per segment | 
| **Window Overlap** | `50%` | 100-sample stride between successive windows | 

## 📊 Performance Evaluation

The models were evaluated using an **80/20 train-test split** stratified across both gait conditions:

| **Model** | **Accuracy** | **Precision** | **Recall** | **F1 Score** | 
| **Logistic Regression** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | 
| **Gaussian Naive Bayes** | **98.00%** | **98.00%** | **98.00%** | **98.00%** | 
| **K-Nearest Neighbors (K=3)** | **92.00%** | **92.00%** | **92.00%** | **92.00%** | 

> **Key Takeaway:** The high dimensional feature space generated from multi-domain signal characteristics allows linear classifiers like **Logistic Regression** to achieve complete separation between normal and abnormal gait states.

## 🚀 Getting Started

### 1. Requirements

Make sure you have the following packages installed:

```
pip install numpy pandas matplotlib scipy scikit-learn

```

### 2. Dataset Setup

Ensure the marker trajectory CSV files are present in your execution environment:

* `Data set_GP1_0.7_marker.csv` (Normal-like Gait)

* `Data set_GP1_0.6_marker.csv` (Abnormal-like Gait)

### 3. Execution

Run the script or Google Colab notebook to execute the end-to-end processing pipeline, output signal plots, train the classifiers, and report matrix metrics.