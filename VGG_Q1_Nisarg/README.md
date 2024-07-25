# Facial Expression Recognition with Streamlit

This repository contains a Streamlit application that uses a pre-trained TensorFlow model to recognize facial expressions in uploaded images. The model predicts one of six facial expressions: Ahegao, Angry, Happy, Neutral, Sad, and Surprise.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project is a web application built with Streamlit that allows users to upload an image and get predictions for facial expressions using a pre-trained TensorFlow model.

## Features

- Upload an image and get real-time predictions for facial expressions.
- Displays the uploaded image along with the predicted expression and associated probabilities.
- Easy-to-use interface built with Streamlit.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.6 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/facial-expression-recognition.git
    cd facial-expression-recognition
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Place the pre-trained model file (`nisarg_model.h5`) in the root directory of the project.

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501` to access the application.

3. Upload an image and click the "OK" button to get the predicted facial expression.

## Project Structure


## Model Training

The model used in this application was trained using the following approach:

- A base model of EfficientNetB0 pre-trained on ImageNet was used.
- Custom layers were added on top of the base model to fine-tune it for facial expression recognition.
- The model was trained on a dataset with the following preprocessing steps:
  - Gaussian blur was applied to each image.
  - Images were resized to 224x224 pixels.
  - Data augmentation techniques such as rotation, width/height shift, and horizontal flip were applied.

## Contributing

Contributions are welcome! Please fork this repository and open a pull request to add improvements or fixes.

