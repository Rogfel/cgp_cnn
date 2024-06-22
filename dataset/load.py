# https://www.tensorflow.org/tutorials/load_data/images
import tensorflow as tf


PATH_DATASET = 'dataset/PetImages/'
BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180

def from_directory():
    train_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_DATASET,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

    test_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_DATASET,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

    return train_ds, test_ds



