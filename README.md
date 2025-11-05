# HopeScan: AI-Powered Mammogram Classification API

HopeScan is a deep learning-based prediction engine designed to classify mammogram images as either 'Benign' or 'Malignant'.

The core of the project is a Convolutional Neural Network (CNN) built with PyTorch, based on the DenseNet architecture. For 'Malignant' predictions, the API also provides an explainable AI (XAI) visualization using Grad-CAM, which generates a heatmap to highlight the regions of the image that were most influential in the model's decision.

## Features

-   **AI-Powered Prediction**: Classifies mammogram images into 'Benign' or 'Malignant' categories.
-   **Explainable AI (XAI)**: Generates a Grad-CAM heatmap overlay on the original image for malignant predictions, providing insight into the model's focus.
-   **RESTful API**: The model is served via a Flask API, allowing any application to request predictions via a simple HTTP POST request.

## Dataset
https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

## Project Structure
HOPESCAN_PROJECT/
├── app.py
├── data/
│   ├── processed_test_data.csv
│   └── processed_train_data.csv
├── model/
│   └── saved_model/
│       └── best_model_checkpoint.pth
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_data_visualization.ipynb
│   └── 04_augmentation_training.ipynb
├── requirements.txt
└── venv/
    ├── ... (Virtual environment files)

## Technology Stack

-   **Machine Learning**: PyTorch, Torchvision
-   **API Framework**: Python, Flask
-   **Data Science & Preprocessing**: Pandas, NumPy, OpenCV, Pillow

## API app.py

This repository contains the official machine learning prediction engine for the HopeScan project. It exposes a simple RESTful API built with Flask to classify mammogram images as 'Benign' or 'Malignant' using a pre-trained PyTorch model.

For malignant predictions, the API also provides an explainable AI (XAI) visualization using Grad-CAM, which generates a heatmap to highlight the regions of the image that were most influential in the model's decision.

### Features

-   **AI-Powered Prediction**: Classifies mammogram images using a DenseNet CNN architecture.
-   **Explainable AI (XAI)**: Generates a Grad-CAM heatmap for malignant predictions.
-   **RESTful API**: Lightweight, scalable, and easy to integrate via a simple HTTP POST request.

### Predict Image

-   **Endpoint**: `/predict`
-   **Method**: `POST`
-   **Description**: Accepts an image file and returns a prediction.
-   **Body**: `form-data` with a key named `file` and the value being the mammogram image file.


## Getting Started

Follow these instructions to set up and run the API server on your local machine.

### Prerequisites

-   Python 3.11 or later
-   `pip` and `venv` (usually included with Python)

### Installation

1.  **Clone the repository:**
  ```
  git clone https://github.com/your-username/hopescan-api.git
  cd hopescan-api
  ```

2.  **Create and activate a Python virtual environment:**
  This isolates project dependencies.
  ```
  # Create the virtual environment
  python -m venv venv

  # Activate the environment (on Windows)
  venv\Scripts\activate
  
  # On macOS/Linux, use: source venv/bin/activate
  ```

3.  **Install the required libraries:**
  Use the `requirements.txt` file to install all necessary dependencies.
  ```
  pip install -r requirements.txt
  ```

### Running the Server

With the virtual environment active, start the Flask server:


