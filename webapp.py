from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from flask_cors import CORS
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='static')
CORS(app)

# app = Flask(__name__, static_folder='static')

# Load the trained model
model = tf.keras.models.load_model('hand_written.keras')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to match the model's input
    input_image = np.array(image) / 255.0  # Normalize the image

    input_image = input_image.reshape(1, 28, 28, 1)  # Reshape for model prediction
    
    prediction = model.predict(input_image)
    
    return jsonify({'prediction': prediction.argmax(axis=-1).tolist()})  # Send back the predicted digit


# Serve the HTML frontend
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
