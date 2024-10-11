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

    # Handle transparency (if any)
    if image.mode == 'RGBA':
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # Remove transparency
        image = background

    # Convert to grayscale and resize
    grayscale_image = image.convert('L').resize((28, 28), Image.Resampling.LANCZOS)
    binary_image = grayscale_image.point(lambda p: p > 128 and 255)  # Binarization

    # Prepare image for model input
    input_image = np.invert(np.array([binary_image]))  # Invert colors for model input
    input_image = input_image.astype('float32') / 255.0  # Normalize

    # Make predictions
    prediction = model.predict(input_image)

    # Log the prediction to check output
    print("Model Prediction:", prediction)

    # Return the path of the heatmap so it can be viewed on the webpage
    return jsonify({
        'prediction': prediction.argmax(axis=-1).tolist(),
        'confidence_scores': prediction[0].tolist()  # Ensure correct extraction of scores
    })

# Serve the HTML frontend
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)