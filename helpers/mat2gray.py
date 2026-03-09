from typing import Any
import numpy as np


def mat2gray(img: Any):
    """mat2gray."""
    min_v = img.min()
    max_v = img.max()
    if max_v - min_v < 1e-10:
        return np.zeros_like(img)
    return (img - min_v) / (max_v - min_v)
