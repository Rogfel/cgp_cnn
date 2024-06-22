from feature_extractions.cnn import conv2D, maxPool2D, flatten

     
def sequency(images_batch):
    # x = rescaling(image=image)
    x = conv2D(image=images_batch, filters=11, kernel_size=4)
    x = maxPool2D(image=x, pool_sizes=3, strides=2)

    x = conv2D(image=x, filters=5)
    x = maxPool2D(image=x, pool_sizes=3, strides=2)

    x = conv2D(image=x, filters=3)
    x = conv2D(image=x, filters=3)
    x = conv2D(image=x, filters=3)

    x = maxPool2D(image=x, pool_sizes=3, strides=2)

    return flatten(image=x)


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
    
