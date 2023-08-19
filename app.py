from flask import Flask, render_template, request, send_file, redirect
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.train import Checkpoint, latest_checkpoint
from model.cae import CAE
from utils import load_image, parse_np_array_image

app = Flask(__name__)

# Load the CAE model and the latest checkpoint
model = CAE()
checkpoint_path = 'training_models/'  # Replace with the actual path to your model checkpoint
ckpt = Checkpoint(transformer=model)
ckpt.restore(latest_checkpoint(checkpoint_path)).expect_partial()

def get_image_size(image_path):
    return os.path.getsize(image_path)

def convert_bytes(size, unit=None):
    # Fungsi ini akan mengkonversi ukuran dari byte menjadi KB, MB, atau GB
    # unit bisa berisi "KB", "MB", atau "GB" (default adalah None)
    if unit == "KB":
        return size / 1024
    elif unit == "MB":
        return size / (1024 ** 2)
    elif unit == "GB":
        return size / (1024 ** 3)
    else:
        return size

# Function to compress the image and save it to the server
def compress_and_save_image(image, image_path):
    compressed_image = compress(model, image)
    compressed_image_array = compressed_image[0].numpy()
    compressed_image_array = (compressed_image_array * 255.0).astype(np.uint8)
    compressed_image_pil = Image.fromarray(compressed_image_array)
    compressed_image_pil.save(image_path)

# Function to compress the image
def compress(model, image):
    image_array = np.array(image)
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    return image_tensor

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if an image file was uploaded
        if 'image' not in request.files:
            return "No image file found."

        image = request.files['image']
        if image.filename == '':
            return "No selected image."

        # Save the uploaded image to a temporary location
        temp_image_path = 'static/temp_image.jpg'
        image.save(temp_image_path)

        # Load the image from the temporary location
        image = load_image(temp_image_path)

        # Process the image (compression)
        compress_and_save_image(image, 'static/compressed_image.jpg')

        # Get image sizes
        original_image_size = get_image_size(temp_image_path)
        compressed_image_size = get_image_size('static/compressed_image.jpg')

        # Convert sizes to KB
        original_image_size = convert_bytes(original_image_size, unit="KB")
        compressed_image_size = convert_bytes(compressed_image_size, unit="KB")

        # Paths for displaying images on the page
        original_image = 'static/temp_image.jpg'
        compressed_image = 'static/compressed_image.jpg'

        return render_template('index.html', original_image=original_image, compressed_image=compressed_image,
                               original_image_size=original_image_size, compressed_image_size=compressed_image_size)

    return render_template('index.html', original_image=None, compressed_image=None)


if __name__ == '__main__':
   port = int(os.environ.get('PORT', 5000))  # Gunakan 5000 sebagai port default
    app.run(host="0.0.0.0", port=port)
