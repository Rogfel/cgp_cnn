from dataset import load
from feature_extractions import cgp
from feature_extractions.vision import vision_functions
from classifications import DT


# load dataset
print('*****1/4 Load images dataset')

load.PATH_DATASET = 'dataset/PetImages/'
images, labels, classes = load.data(data_type='train')

# feature extraction
print('*****2/4 Init feature extraction')
# Initialize CGP
cgp.INPUT_SHAPE = (load.IMG_HEIGHT, load.IMG_WIDTH, 3)  # RGB images
cgp.FUNCTIONS = vision_functions()
cgp.N_NODES = cgp.get_n_nodes()
# Evolve CGP
best_genome, best_fitness = cgp.evolve(images, labels,
                                       n_generations=10,
                                       population_size=3,
                                       eval_model=DT.classification_model(),
                                       mutation_rate=0.3)

print(f"Training completed. Best fitness: {best_fitness}\n")
print(f"best genome: {best_genome}")