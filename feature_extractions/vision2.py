import numpy as np
from dataclasses import dataclass
from typing import List, Callable
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'  # Or "jax" or "torch"!
import keras


def reshape(image: np.ndarray) -> np.ndarray:
    """
    Reshape input image to be compatible with Keras Conv2D layers
    
    Args:
        image: Input image with shape (height, width) or (height, width, channels)
        
    Returns:
        np.ndarray with shape (1, height, width, 1) or (1, height, width, channels)
    """
    # Check current shape
    if len(image.shape) == 2:
        # Add batch and channel dimensions
        return image.reshape(1, image.shape[0], image.shape[1], 1)
    elif len(image.shape) == 3:
        # Add only batch dimension
        return image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    elif len(image.shape) == 4:
        # Already in correct shape
        return image
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

@dataclass
class NodeFunction:
    func: Callable
    name: str
    n_inputs: int


def vision_functions() -> List[NodeFunction]:
        """Initialize the set of possible node functions"""
        
        def rescaling(image):
            # argument 0 image
            return keras.layers.Rescaling(1./255)(image)

        def __conv2D(image, filters=32, kernel_size=3):
            return keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(image)
        
        def conv2D32_3(image):
              return __conv2D(image=image, filters=32, kernel_size=3)
        
        def conv2D32_5(image):
              return __conv2D(image=image, filters=32, kernel_size=5)
        
        def conv2D64_3(image):
              return __conv2D(image=image, filters=64, kernel_size=3)
        
        def conv2D64_5(image):
              return __conv2D(image=image, filters=64, kernel_size=5)
        
        def conv2D128_3(image):
              return __conv2D(image=image, filters=128, kernel_size=3)
        
        def conv2D128_5(image):
              return __conv2D(image=image, filters=128, kernel_size=5)

        def maxPool2D(image):
            pool_sizes=2
            strides=2
            return keras.layers.MaxPool2D((pool_sizes, pool_sizes), (strides, strides),
                                        padding="valid")(image)

        def avgPool2D(image):
            pool_sizes=2
            strides=2
            return keras.layers.AveragePooling2D((pool_sizes, pool_sizes), (strides, strides),
                                                padding="valid")(image)
        def concatenate(image0, image1):
            return keras.layers.concatenate([image0, image1])

        def summation(image0, image1):
            return keras.layers.add([image0, image1])

        def resnet(image):
            filters=32
            kernel_size=3
            x = __conv2D(image=image, filters=filters, kernel_size=kernel_size)
            x = keras.layers.BatchNormalization()(image)
            x = summation(image0=image, image1=x)
            return keras.layers.ReLU()(x)

        return [
            # NodeFunction(rescaling, "rescaling", 1),
            NodeFunction(conv2D32_3, "conv2D32_3", 1),
            NodeFunction(conv2D32_5, "conv2D32_5", 1),
            NodeFunction(conv2D64_3, "conv2D64_3", 1),
            NodeFunction(conv2D64_5, "conv2D64_5", 1),
            # NodeFunction(conv2D128_3, "conv2D128_3", 1),
            # NodeFunction(conv2D128_5, "conv2D128_5", 1),
            NodeFunction(maxPool2D, "maxPool2D", 1),
            NodeFunction(avgPool2D, "avgPool2D", 1),
            # NodeFunction(concatenate, "concatenate", 2),
            # NodeFunction(summation, "summation", 2),
            NodeFunction(resnet, "resnet", 1),
            # NodeFunction(flatten, "flatten", 1)
        ]

def flatten(image):
    return keras.layers.Flatten()(image)