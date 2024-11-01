import numpy as np
from dataset import load
from feature_extractions import cgp
from feature_extractions.vision import vision_functions
from classifications import random_forests


# load dataset
print('*****1/4 Load images dataset')
# load.PATH_DATASET = 'dataset/PetImages/'
train_data, train_target, test_data, test_target = load.data()

# feature extraction
print('*****2/4 Init feature extraction')
# Initialize CGP
cgp.INPUT_SHAPE = (load.IMG_HEIGHT, load.IMG_WIDTH, 3)  # RGB images
cgp.FUNCTIONS = vision_functions()
cgp.N_NODES = cgp.get_n_nodes()
# Evolve CGP
best_genome, best_fitness = cgp.evolve(train_data[:10], train_target[:10],
                                       test_data[:2], test_target[:2], n_generations=10,
                                       population_size=50, eval_model=random_forests.classification_model())

print(f"Training completed. Best fitness: {best_fitness}")
print(f"best genome: {best_genome}")