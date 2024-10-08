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
import uuid

app = Flask(__name__, static_folder='static')
CORS(app)

# app = Flask(__name__, static_folder='static')

# Load the trained model
model = tf.keras.models.load_model('hand_written2.0.keras')

from PIL import Image

# def preprocess_image(input_image):
#     print(f"Original image shape: {input_image.shape}")
    
#     if len(input_image.shape) == 2:  # If it's a grayscale image
#         input_image = np.expand_dims(input_image, axis=-1)  # Add channel dimension
#         print(f"After adding channel dimension: {input_image.shape}")
    
#     input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
#     print(f"After adding batch dimension: {input_image.shape}")
    
#     # Remove the extra batch dimension if it exists
#     if len(input_image.shape) == 5:
#         input_image = np.squeeze(input_image, axis=0)
#         print(f"After removing extra batch dimension: {input_image.shape}")
    
#     return input_image

# def get_grad_cam_heatmap(model, input_image, predicted_class):
#     # Call the model with a dummy input to ensure layers are initialized
#     _ = model.predict(tf.zeros((1, 28, 28, 1)))  # Dummy input
    
#     # Check and debug available layers
#     for layer in model.layers:
#         print(layer.name)  # Print layer names to confirm 'conv2d_1' exists

#     # Create a new model for Grad-CAM
#     grad_model = tf.keras.models.Model(
#         inputs=model.inputs,
#         outputs=[model.get_layer('conv2d_1').output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         # Forward pass
#         conv_output, predictions = grad_model(input_image)
#         loss = predictions[:, predicted_class]
    
#     # Gradient of the predicted class wrt conv layer output
#     grads = tape.gradient(loss, conv_output)
    
#     # Compute the channel-wise mean of the gradients
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
#     # Multiply each channel in the feature map array by 'how important this channel is' with respect to the class
#     conv_output = conv_output[0]
#     heatmap = conv_output @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)

#     # Apply ReLU to discard negative values
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

#     return heatmap.numpy()




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

    plt.imshow(binary_image, cmap='gray')  # Display the original image
    plt.axis('off')
    plt.savefig('binary_image.png', bbox_inches='tight')  # Save the image to a file

    # Save the heatmap image with a unique filename
    heatmap_filename = f'heatmap_{uuid.uuid4().hex}.png'
    heatmap_path = os.path.join(app.static_folder, heatmap_filename)
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)

    # Return the path of the heatmap so it can be viewed on the webpage
    return jsonify({
        'prediction': predicted_class,
        'confidence_scores': prediction[0].tolist(),
        'heatmap_url': f'static/{heatmap_filename}'  # Return the unique heatmap image path
    })

# Serve the HTML frontend
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)