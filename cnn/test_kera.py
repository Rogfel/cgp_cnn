import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

import tensorflow as tf
import keras_cv
import tensorflow_datasets as tfds
import keras

# Create a preprocessing pipeline with augmentations
BATCH_SIZE = 16
NUM_CLASSES = 3
augmenter = keras_cv.layers.Augmenter(
    [
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandAugment(value_range=(0, 255)),
        keras_cv.layers.CutMix(),
    ],
)

def preprocess_data(images, labels, augment=False):
    labels = tf.one_hot(labels, NUM_CLASSES)
    inputs = {"images": images, "labels": labels}
    outputs = inputs
    if augment:
        outputs = augmenter(outputs)
    return outputs['images'], outputs['labels']

train_dataset, test_dataset = tfds.load(
    'rock_paper_scissors',
    as_supervised=True,
    split=['train', 'test'],
)
train_dataset = train_dataset.batch(BATCH_SIZE).map(
    lambda x, y: preprocess_data(x, y, augment=True),
        num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).map(
    preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE)

# Create a model using a pretrained backbone
backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(
    "efficientnetv2_b0_imagenet"
)
model = keras_cv.models.ImageClassifier(
    backbone=backbone,
    num_classes=NUM_CLASSES,
    activation="softmax",
)
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    metrics=['accuracy']
)

# Train your model
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=8,
)
