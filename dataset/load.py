# https://www.tensorflow.org/tutorials/load_data/images
import numpy as np
import tensorflow as tf


PATH_DATASET = 'dataset/PetImages/'
BATCH_SIZE = 32
IMG_HEIGHT = 64
IMG_WIDTH = 64

def data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
                    PATH_DATASET,
                    validation_split=0.8,
                    subset="training",
                    seed=123,
                    image_size=(IMG_HEIGHT, IMG_WIDTH),
                    batch_size=BATCH_SIZE)

    test_ds = tf.keras.utils.image_dataset_from_directory(
                    PATH_DATASET,
                    validation_split=0.8,
                    subset="validation",
                    seed=123,
                    image_size=(IMG_HEIGHT, IMG_WIDTH),
                    batch_size=BATCH_SIZE)
    
    def transform_data(data_target):
        data = None
        for batch in data_target:
            if data is None:
                data = np.array(batch[0])
                target = np.array(batch[1])
            else:
                data = np.concatenate((data, np.array(batch[0])), axis=0)
                target = np.concatenate((target, np.array(batch[1])), axis=0)
        return data, target

    train_data, train_target = transform_data(train_ds)
    test_data, test_target = transform_data(test_ds)

    return train_data, train_target, test_data, test_target


