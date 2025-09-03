# Animal Recognition App

## Overview

This is an AI-powered animal recognition application built with Streamlit and TensorFlow. The app allows users to upload photos of animals and uses a pre-trained MobileNetV2 model to identify the animal species. The system leverages ImageNet classification capabilities, mapping the model's predictions to common animal categories that users are likely to encounter.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Interface**: Single-page application with a clean, user-friendly interface
- **File Upload Component**: Supports multiple image formats (JPG, JPEG, PNG, BMP, GIF)
- **Two-Column Layout**: Displays uploaded image alongside prediction results
- **Caching Strategy**: Uses Streamlit's `@st.cache_resource` to cache the ML model for performance

### Backend Architecture
- **Modular Design**: Separated into three main components:
  - `app.py`: Streamlit frontend and user interaction
  - `animal_classifier.py`: Core ML classification logic
  - `model_utils.py`: Image preprocessing and class mapping utilities
- **Object-Oriented Pattern**: AnimalClassifier class encapsulates model loading and prediction logic
- **Error Handling**: Comprehensive exception handling with logging throughout the application

### Machine Learning Pipeline
- **Pre-trained Model**: Uses MobileNetV2 with ImageNet weights for transfer learning
- **Image Preprocessing**: Standardized pipeline including:
  - RGB conversion
  - Resizing to 224x224 pixels
  - Normalization and scaling for MobileNetV2 input requirements
- **Classification Mapping**: Custom mapping from ImageNet classes to user-friendly animal names
- **Prediction Output**: Returns animal name, confidence percentage, and top predictions list

### Data Processing
- **Image Handling**: PIL (Python Imaging Library) for image manipulation
- **NumPy Arrays**: Efficient array operations for image preprocessing
- **TensorFlow Integration**: Native TensorFlow preprocessing functions for model compatibility

## External Dependencies

### Machine Learning Framework
- **TensorFlow**: Core ML framework providing MobileNetV2 model and preprocessing utilities
- **ImageNet Weights**: Pre-trained weights for transfer learning (downloaded automatically)

### Web Framework
- **Streamlit**: Web application framework for creating the user interface

### Image Processing
- **PIL (Pillow)**: Python Imaging Library for image manipulation and format conversion
- **NumPy**: Numerical computing library for array operations

### Development Tools
- **Python Logging**: Built-in logging module for error tracking and debugging

### Model Specifications
- **MobileNetV2**: Lightweight convolutional neural network optimized for mobile and embedded devices
- **Input Requirements**: 224x224x3 RGB images
- **ImageNet Classification**: 1000-class classification system with animal subset mapping