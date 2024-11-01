import numpy as np
from scipy import ndimage
import cv2
from dataclasses import dataclass
import random
from typing import List, Callable


@dataclass
class NodeFunction:
    func: Callable
    name: str
    n_inputs: int


def vision_functions() -> List[NodeFunction]:
        """Initialize the set of possible node functions"""
        
        def conv3x3(x: np.ndarray, param: float) -> np.ndarray:
            # Handle each channel separately
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                kernel = np.array([[param, param, param],
                                 [param, 1.0, param],
                                 [param, param, param]])
                for c in range(x.shape[2]):
                    result[..., c] = ndimage.convolve(np.clip(x[..., c], -1e6, 1e6), kernel, mode='reflect')
                return np.clip(result, -1e6, 1e6)
            
            kernel = np.array([[param, param, param],
                                [param, 1.0, param],
                                [param, param, param]])
            result = ndimage.convolve(np.clip(x, -1e6, 1e6), kernel, mode='reflect')
            return np.clip(result, -1e6, 1e6)
            
        def max_pool(x: np.ndarray, param: float) -> np.ndarray:
            size = max(2, int(abs(param * 3)))
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    result[..., c] = ndimage.maximum_filter(np.clip(x, -1e6, 1e6), size=size)
                return np.clip(result, -1e6, 1e6)
            return np.clip(ndimage.maximum_filter(np.clip(x, -1e6, 1e6), size=size), -1e6, 1e6)
            
        def avg_pool(x: np.ndarray, param: float) -> np.ndarray:
            size = max(2, int(abs(param * 3)))
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    result[..., c] = ndimage.uniform_filter(np.clip(x, -1e6, 1e6), size=size)
                return result
            return np.clip(ndimage.uniform_filter(np.clip(x, -1e6, 1e6), size=size), -1e6, 1e6)
        
        def gaussian_blur(x: np.ndarray, param: float) -> np.ndarray:
            sigma = abs(param)
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    result[..., c] = ndimage.gaussian_filter(np.clip(x, -1e6, 1e6), sigma=sigma)
                return np.clip(result, -1e6, 1e6)
            return np.clip(ndimage.gaussian_filter(np.clip(x, -1e6, 1e6), sigma=sigma), -1e6, 1e6)
            
        def sobel_x(x: np.ndarray, param: float) -> np.ndarray:
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    result[..., c] = ndimage.sobel(np.clip(x, -1e6, 1e6), axis=0) * param
                return result
            return ndimage.sobel(x, axis=0) * param
            
        def sobel_y(x: np.ndarray, param: float) -> np.ndarray:
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    result[..., c] = ndimage.sobel(np.clip(x, -1e6, 1e6), axis=1) * param
                return result
            return ndimage.sobel(x, axis=1) * param
            
        def threshold(x: np.ndarray, param: float) -> np.ndarray:
            return np.where(x > param, 1.0, 0.0)
            
        def normalize(x: np.ndarray, param: float) -> np.ndarray:
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    channel = x[..., c]
                    min_val = np.min(channel)
                    max_val = np.max(channel)
                    if max_val > min_val:
                        result[..., c] = (channel - min_val) / (max_val - min_val)
                    else:
                        result[..., c] = channel
                return result
            else:
                min_val = np.min(x)
                max_val = np.max(x)
                if max_val > min_val:
                    return (x - min_val) / (max_val - min_val)
                return x
            
        def add(x1: np.ndarray, x2: np.ndarray, param: float) -> np.ndarray:
            # Ensure the arrays have the same shape
            if x1.shape != x2.shape:
                x2 = cv2.resize(x2, (x1.shape[1], x1.shape[0]))
                if len(x1.shape) == 3 and len(x2.shape) == 2:
                    x2 = np.expand_dims(x2, axis=-1)
                    x2 = np.repeat(x2, x1.shape[2], axis=-1)
            return x1 + x2 * param
            
        def multiply(x1: np.ndarray, x2: np.ndarray, param: float) -> np.ndarray:
            # Ensure the arrays have the same shape
            if x1.shape != x2.shape:
                x2 = cv2.resize(x2, (x1.shape[1], x1.shape[0]))
                if len(x1.shape) == 3 and len(x2.shape) == 2:
                    x2 = np.expand_dims(x2, axis=-1)
                    x2 = np.repeat(x2, x1.shape[2], axis=-1)
            return x1 * x2 * param

        return [
            NodeFunction(conv3x3, "conv3x3", 1),
            NodeFunction(max_pool, "max_pool", 1),
            NodeFunction(avg_pool, "avg_pool", 1),
            NodeFunction(gaussian_blur, "gaussian_blur", 1),
            NodeFunction(sobel_x, "sobel_x", 1),
            NodeFunction(sobel_y, "sobel_y", 1),
            NodeFunction(threshold, "threshold", 1),
            NodeFunction(normalize, "normalize", 1),
            NodeFunction(add, "add", 2),
            NodeFunction(multiply, "multiply", 2)
        ]