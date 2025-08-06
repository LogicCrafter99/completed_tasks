import cv2
import io
import base64
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

IMG_SIZE = 64


def img_preprocessing(image_base_64):
    # Decode base64 string to bytes
    image_data = base64.b64decode(image_base_64)

    # Open image from bytes and convert to RGB
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Convert image to NumPy array
    image_np = np.array(image)

    # Convert RGB image to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Apply Otsu's thresholding for binary segmentation
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Create kernel for morphological operation
    kernel = np.ones((5, 5), np.uint8)

    # Apply morphological closing to remove noise
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Resize image to IMG_SIZE x IMG_SIZE (64x64)
    resized = cv2.resize(morph, (IMG_SIZE, IMG_SIZE))

    # Normalize pixel values to [0, 1]
    normalized = resized / 255.0

    # Reshape image to add batch size and channel dimensions
    reshaped = np.reshape(normalized, (1, IMG_SIZE, IMG_SIZE, 1))

    return reshaped
