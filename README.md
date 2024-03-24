# CNN-LSTM Storm Identification

## Description

This project utilizes a Convolutional Neural Network (CNN) combined with Long Short-Term Memory (LSTM) units to predict and identify storm patterns in image sequences. The model is trained to analyze sequences of images and predict future frames, potentially identifying the development of storms.

## Installation

To install the package, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/<your-username>/CNN_LSTM_Storm_Identification.git
    ```
2. Navigate to the cloned directory:
    ```bash
    cd CNN_LSTM_Storm_Identification
    ```
3. Install the package:
    ```bash
    pip install .
    ```

Ensure you have Python 3.7 or later installed on your machine.

## Usage

Here's a quick example to get you started with using the CNN-LSTM model for storm identification:

1. **Download the Pretrained Model**

   Before running the inference, ensure that you've downloaded the pretrained model using the `download_model_from_drive` function provided in the `inference.py` script.

2. **Prepare Your Data**

   Your image sequence should be organized and named appropriately. This model expects sequences of grayscale images resized to 128x128 pixels.

3. **Perform Inference**

   ```python
   from cnn_lstm.inference import load_model, preprocess_input, perform_inference

   # Load the pretrained model
   model_path = 'path_to_your_model.pth'
   model = load_model(model_path)

   # Preprocess your input data
   input_data = preprocess_input('path_to_your_image_sequence')

   # Perform inference
   predicted_images = perform_inference(model, input_data)

   # Save or display your results
   save_images(images)
   ```
