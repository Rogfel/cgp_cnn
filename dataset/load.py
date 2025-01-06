import os
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


PATH_DATASET = 'dataset/PetImages/'
BATCH_SIZE = 32
IMG_HEIGHT = 64
IMG_WIDTH = 64


def data(data_type='train'):

    # Define transformations
    if data_type == 'train' or data_type == 'val':
        data_transforms = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return None

    image_datasets = datasets.ImageFolder(os.path.join(PATH_DATASET, data_type),
                                          data_transforms)
    dataloader = DataLoader(image_datasets, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    class_names = image_datasets.classes

    images_list = []
    labels_list = []

    for images_batch, labels_batch in dataloader:
        images_list.append(images_batch.numpy())  # Convert to NumPy immediately
        labels_list.append(labels_batch.numpy())

    images_array = np.concatenate(images_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    return images_array, labels_array, class_names
