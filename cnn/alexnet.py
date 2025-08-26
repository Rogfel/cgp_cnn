# https://github.com/visionatseecs/keras-starter/blob/main/keras_alexnet.ipynb
from feature_extractions import conv_layers as cnn

     
def sequency(images_batch):
    # x = rescaling(image=image)
    x = cnn.Layers.exec('conv2D',[images_batch, 11, 4])
    x = cnn.Layers.exec('maxPool2D', [x, 3, 2])

    x = cnn.Layers.exec('conv2D', [x, 5, 3])
    x = cnn.Layers.exec('maxPool2D', [x, 3, 2])

    x = cnn.Layers.exec('conv2D', [x, 3, 3])
    x = cnn.Layers.exec('conv2D', [x, 3, 3])
    x = cnn.Layers.exec('conv2D', [x, 3, 3])

    x = cnn.Layers.exec('maxPool2D', [x, 3, 2])

    return cnn.Layers.exec('flatten', [x])


if __name__ == '__main__':
    from dataset import load
    load.PATH_DATASET = 'dataset/PetImages/'

    
    train_ds, test_ds = load.from_directory()
    
    # for img in glob.glob('dataset/PetImages/Dog/*'):
    #     print(img)
    #     image_file = tf.io.read_file(img)
    #     image = tf.image.decode_image(image_file)

    # try:
    #     for val in val_ds:
    #         print((sequency(image=val[0]), val[1]))
    # except StopIteration:
    #     print("End of dataset reached.")

    # Create an iterator
    iterator = iter(test_ds)

    # Read data from the iterator
    try:
        while True:
            batch = next(iterator)
            print((sequency(image=batch[0]), batch[1]))
            # print(batch.numpy())
    except StopIteration:
        print("End of dataset reached.")
    
