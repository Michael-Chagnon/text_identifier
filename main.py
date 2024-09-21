from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('handwritten.model')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to match the input shape
    img = np.array(img) / 255.0  # Normalize
    img = img.reshape(1, 28, 28, 1)  # Reshape to match model input
    
    # Predict the digit
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    return jsonify({'digit': int(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)
