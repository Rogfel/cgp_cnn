import numpy as np
import tensorflow as tf
from dataset import load
from feature_extractions.test_cnn import sequency
from feature_simplifications import tgp
from classifications import random_forests
from classifications import evaluation


# load dataset
print('*****1/4 Load dataset')
load.PATH_DATASET = 'dataset/PetImages/'
train_ds, test_ds = load.from_directory()

# feature extraction
print('*****2/4 Init feature extraction')
train_extraction_data, train_extraction_target = None, None
for batch in train_ds:
    if train_extraction_data is None:
        train_extraction_data = np.array(sequency(batch[0]))
        train_extraction_target = np.array(batch[1])
    else:
        train_extraction_data = np.concatenate((train_extraction_data,
                                                np.array(sequency(batch[0]))), axis=0)
        train_extraction_target = np.concatenate((train_extraction_target,
                                                  np.array(batch[1])), axis=0)
print('End training dataset extraction')
test_extraction_data, test_extraction_target = None, None
for batch in train_ds:
    if test_extraction_data is None:
        test_extraction_data = np.array(sequency(batch[0]))
        test_extraction_target = np.array(batch[1])
    else:
        test_extraction_data = np.concatenate((test_extraction_data,
                                                np.array(sequency(batch[0]))), axis=0)
        test_extraction_target = np.concatenate((test_extraction_target,
                                                  np.array(batch[1])), axis=0)
print('End testing dataset extraction')

# feature simplification
print('***** 3/4 Init TGP transformation')
GP = tgp.transformer()
GP.fit(train_extraction_data, train_extraction_target)

# all_dataset = np.concatenate((train_extraction_data, test_extraction_data), axis=0)
# all_dataset_target = np.concatenate((train_extraction_target, test_extraction_target), axis=0)

new_train_data = np.hstack((train_extraction_data, GP.transform(train_extraction_data)))
new_test_data = np.hstack((test_extraction_data, GP.transform(test_extraction_data)))

# Classification
print('***** 4/4 Init Classification')
RF = random_forests.classification_model()
RF.fit(new_train_data, train_extraction_target)
print(evaluation.roc_auc_score(new_test_data, test_extraction_target))

# RF.fit(train_extraction_data, train_extraction_target)
# print(evaluation.roc_auc_score(test_extraction_data, test_extraction_target))