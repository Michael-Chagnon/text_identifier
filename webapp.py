from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import io
from flask_cors import CORS
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow.keras.backend as K

app = Flask(__name__, static_folder='static')
CORS(app)

# app = Flask(__name__, static_folder='static')

# Load the trained model
model = tf.keras.models.load_model('hand_written2.0.keras')

from PIL import Image

def get_grad_cam_heatmap(model, input_image, predicted_class):
    # Ensure the model runs at least once to initialize the outputs
    _ = model.predict(input_image)

    # Create the Grad-CAM model from the conv2d_1 layer to the output
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer('conv2d').output, model.output]  # Change 'conv2d_1' based on the layer you want
    )

    with tf.GradientTape() as tape:
        # Forward pass through the grad_model
        conv_outputs, predictions = grad_model(input_image)
        loss = predictions[:, predicted_class]  # Loss for the predicted class

    # Calculate the gradients of the loss with respect to the conv layer outputs
    grads = tape.gradient(loss, conv_outputs)

    # Pool the gradients over all the axes
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each feature map in the convolutional layer output by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Apply ReLU to remove negative values and normalize the heatmap
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap)

    return heatmap.numpy()


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
    input_image = input_image.reshape((1, 28, 28, 1))  # Reshape for model

    # Make predictions
    prediction = model.predict(input_image)

    # Get the class with the highest prediction score
    predicted_class = np.argmax(prediction, axis=-1)[0]

    # Generate Grad-CAM heatmap
    heatmap = get_grad_cam_heatmap(model, input_image, predicted_class)

    # Create the heatmap visualization
    plt.imshow(binary_image, cmap='gray')  # Display the original image
    plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Overlay the heatmap
    plt.axis('off')

    # Save the heatmap image
    heatmap_path = 'static/heatmap.png'
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)

    return jsonify({
        'prediction': predicted_class,
        'confidence_scores': prediction[0].tolist(),
        'heatmap_url': heatmap_path  # Return the heatmap image path
    })


# Serve the HTML frontend
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)