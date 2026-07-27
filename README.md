# 🫁 Pneumonia Detection App

## Overview
A machine learning web app that detects pneumonia from chest X-ray images. Users upload an X-ray and a CNN model predicts Pneumonia vs. Normal, along with a confidence score.

This project demonstrates end-to-end ML deployment: model training, backend inference, and a frontend UI — for educational purposes only, not clinical use.

## Features
- Upload chest X-ray images for analysis
- Binary classification: Pneumonia vs. Normal
- Adjustable decision threshold
- Trained Convolutional Neural Network (CNN)
- Flask-based backend for model inference

## Model
- Architecture: Convolutional Neural Network (CNN)
- Framework: TensorFlow / Keras
- Input: Chest X-ray images (resized to 224x224 & normalized)
- Output: Probability score + predicted class

## Tech Stack
**Frontend:** HTML, CSS
**Backend:** Python, Flask
**ML:** TensorFlow / Keras, NumPy, Pillow
