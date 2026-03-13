"""
utils/image_utils.py
Image preprocessing helper for the CNN disease detection model.
Resizes to 224×224, normalises pixel values to [0, 1], adds batch dimension.
"""

import io
import numpy as np
from PIL import Image


def preprocess_image(image_bytes: bytes) -> "np.ndarray":
    """
    Preprocess raw image bytes for MobileNetV2 inference.

    Steps:
    1. Decode bytes with PIL
    2. Convert to RGB (handles grayscale / RGBA inputs)
    3. Resize to 224×224
    4. Normalise pixel values to [0.0, 1.0]
    5. Add batch dimension → shape (1, 224, 224, 3)

    Returns:
        numpy.ndarray of shape (1, 224, 224, 3), dtype float32
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0   # normalise to [0, 1]
    arr = np.expand_dims(arr, axis=0)                # add batch dim
    return arr
