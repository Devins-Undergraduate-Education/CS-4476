import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:
    """
    Compute the mean and the standard deviation of the pixel values in the dataset.

    Note: convert the image in grayscale and then scale to [0,1] before computing
    mean and standard deviation

    Hints: use StandardScalar (check import statement)

    Args:
    -   dir_name: the path of the root dir
    Returns:
    -   mean: mean value of the dataset (np.array containing a scalar value)
    -   std: standard deviation of th dataset (np.array containing a scalar value)
    """

    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################
    image_paths = glob.glob(os.path.join(dir_name, '**', '*.*'), recursive=True)
    scaler = StandardScaler()

    pixel_values = []

    for img_path in image_paths:
        img = Image.open(img_path).convert('L')
        img_array = np.array(img, dtype=np.float32) / 255.0  # scale to [0,1]
        pixel_values.append(img_array.flatten())

    all_pixels = np.concatenate(pixel_values).reshape(-1, 1)

    # Fit the scaler and extract mean/std
    scaler.fit(all_pixels)
    mean = np.array([scaler.mean_[0]])
    std = np.array([np.sqrt(scaler.var_[0])])
    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
