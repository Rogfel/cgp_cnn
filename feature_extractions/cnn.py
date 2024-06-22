# https://keras.io/api/layers/
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ['KERAS_BACKEND'] = 'tensorflow'  # Or "jax" or "torch"!
import keras


def rescaling(image):
    return keras.layers.Rescaling(1./255)(image)

def conv2D(image, filters=32, kernel_size=3):
    return keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(image)

def maxPool2D(image, pool_sizes=2, strides=2):
    return keras.layers.MaxPool2D((pool_sizes, pool_sizes), (strides, strides),
                                  padding="valid")(image)

def avgPool2D(image, pool_sizes=2, strides=2):
    return keras.layers.AveragePooling2D((pool_sizes, pool_sizes), (strides, strides),
                                         padding="valid")(image)

def concatenate(image0, image1):
    return keras.layers.concatenate(axis=1)([image0, image1])

def summation(image0, image1):
    return keras.layers.add(axis=1)([image0, image1])

def flatten(image):
    return keras.layers.Flatten()(image)

def resnet(image, filter=32, kernel_size=3):
    x = conv2D(image=image, filter=filter, kernel_size=kernel_size)
    x = keras.layers.BatchNormalization()(image)
    x = summation(image0=image, image1=x)
    return keras.layers.ReLU()(x)