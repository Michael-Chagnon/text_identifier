from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import io
from flask_cors import CORS
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='static')
CORS(app)

# app = Flask(__name__, static_folder='static')

# Load the trained model
model = tf.keras.models.load_model('hand_written2.0.keras')

from PIL import Image

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    
    # Open the image
    image = Image.open(file.stream)
    image.save('original_image_before_grayscale.png')  # Save original for debugging

    # Handle transparency (if any) before converting to grayscale
    if image.mode == 'RGBA':  # Check if image has alpha channel
        # Create a white background and paste the image onto it (removing transparency)
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # Paste using alpha channel as mask
        image = background

    # Convert to grayscale
    grayscale_image = image.convert('L')  # Convert to grayscale

    # Resize with LANCZOS filter to improve quality
    resized_image = grayscale_image.resize((28, 28), Image.Resampling.LANCZOS)
    resized_image.save('resized_image.png')  # Save resized image for debugging

    # Optionally apply thresholding or binarization to clean up the image
    threshold = 128
    binary_image = resized_image.point(lambda p: p > threshold and 255)  # Binarization
    binary_image.save('binary_image.png')  # Save binary image for debugging

    # Normalize the image
    # input_image = np.array(binary_image) / 255.0  # Normalize
    # input_image = input_image.reshape(1, 28, 28, 1)  # Reshape for model input
    input_image = np.invert(np.array([binary_image]))

    # Predict using the model
    prediction = model.predict(input_image)


    return jsonify({'prediction': prediction.argmax(axis=-1).tolist()})



# Serve the HTML frontend
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)