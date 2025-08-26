from dataset import load
from feature_extractions import cgp
from feature_extractions.vision import vision_functions
from classifications import DT
from classifications import random_forests as RF
from sklearn.model_selection import train_test_split


# load dataset
print('*****1/4 Load images dataset')

load.PATH_DATASET = 'dataset/PetImages/'
images, labels, classes = load.data(data_type='train')

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

# feature extraction
print('*****2/4 Init feature extraction')
# Initialize CGP
cgp.INPUT_SHAPE = (load.IMG_HEIGHT, load.IMG_WIDTH, 3)  # RGB images
cgp.FUNCTIONS = vision_functions()
cgp.N_NODES = cgp.get_n_nodes()
# Evolve CGP
best_genome, best_fitness, val_fitness = cgp.evolve(
    X_train, y_train,
    X_val, y_val,
    n_generations=5,
    population_size=15,
    eval_model=RF.classification_model(),
    mutation_rate=0.1
)

print(f"Training completed.")
print(f"Best training fitness: {best_fitness:.4f}")
print(f"Validation fitness: {val_fitness:.4f}")
print(f"Best genome: {best_genome}")