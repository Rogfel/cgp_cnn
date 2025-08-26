# https://keras.io/api/layers/
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ['KERAS_BACKEND'] = 'tensorflow'  # Or "jax" or "torch"!
import keras

class Layers:

    @classmethod
    def rescaling(cls, arguments):
        # argument 0 image
        image = arguments[0]
        return keras.layers.Rescaling(1./255)(image)

    @classmethod
    def conv2D(cls, image, filters=32, kernel_size=3, padding='valid', activation='relu'):
        return keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation)(image)

    @classmethod
    def maxPool2D(cls, image, pool_sizes=2, strides=2):
        return keras.layers.MaxPool2D((pool_sizes, pool_sizes), (strides, strides),
                                    padding="valid")(image)

    @classmethod
    def avgPool2D(cls, image, pool_sizes=2, strides=2):
        return keras.layers.AveragePooling2D((pool_sizes, pool_sizes), (strides, strides),
                                            padding="valid")(image)
    @classmethod
    def concatenate(cls, image0, image1):
        return keras.layers.Concatenate(axis=1)([image0, image1])

    @classmethod
    def summation(cls, image0, image1):
        return keras.layers.Add()([image0, image1])

    @classmethod
    def flatten(cls, image):
        return keras.layers.Flatten()(image)

    @classmethod
    def resnet(cls, image, filter=32, kernel_size=3):
        x = cls.conv2D(image=image, filter=filter, kernel_size=kernel_size)
        x = keras.layers.BatchNormalization()(image)
        x = cls.summation(image0=image, image1=x)
        return keras.layers.ReLU()(x)


    @classmethod
    def exec(cls, function_name, arguments):
        func = getattr(cls, function_name) 
        if func:
            return func(*arguments)
        else:
            return 'the function does not exist'
        
    @classmethod
    def get_layers_names(cls):
        return [method for method in dir(cls) if not method.startswith("__") and not method.startswith("get") and not method.startswith("exec")]
    

if __name__ == '__main__':
    print(Layers.get_layers_names())