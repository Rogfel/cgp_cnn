import numpy as np
import cv2
from dataclasses import dataclass
import random
from typing import List, Tuple, Type
import torch
import torch.nn.functional as F
from classifications import evaluation


#INPUT_SHAPE: (height, width, channels)
INPUT_SHAPE=None
#N_COLUMNS: Number of columns in CGP grid
N_COLUMNS=20
#N_ROWS: Number of rows in CGP grid
N_ROWS=5
#N_BACK: Number of columns to look back for connections
N_BACK=10
#N_OUTPUTS: Number of features to extract
N_OUTPUTS=64
# Define available functions
FUNCTIONS = None
# Calculate genome length
GENES_PER_NODE=4  # function_id, input1, input2, parameter
# numbers of nodes: N_COLUMNS * N_ROWS (to avoid error, the N_NODES must be calculated in the inicialization process)
N_NODES=None


def get_n_nodes():
    return N_COLUMNS * N_ROWS


def get_genome_length():
    return get_n_nodes() * GENES_PER_NODE + N_OUTPUTS


def create_individual() -> List[float]:
    """Create a random CGP individual"""
    genome = []
    
    # Generate nodes
    for i in range(N_NODES):
        # Function gene
        function_id = random.randint(0, len(FUNCTIONS) - 1)
        genome.append(function_id)
        
        # Input connection genes
        # n_inputs = FUNCTIONS[function_id].n_inputs
        for _ in range(2):  # Always store 2 inputs, even if not used
            x = random.randint(max(0, i - N_BACK), i + 3 - 1)  # +3 for RGB channels
            genome.append(x)
        
        # Parameter gene
        genome.append(random.uniform(-1.0, 1.0))
    
    # Output connection genes
    for _ in range(N_OUTPUTS):
        genome.append(random.randint(3, N_NODES + 2))  # +3 for RGB channels, -1 for 0-based
        
    return genome


def mutate(genome: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Mutate a CGP individual"""
    new_genome = genome.copy()
    
    for i in range(len(new_genome)):
        if random.random() < mutation_rate:
            if i < N_NODES * GENES_PER_NODE:
                node_index = i // GENES_PER_NODE
                gene_type = i % GENES_PER_NODE
                
                if gene_type == 0:  # Function gene
                    new_genome[i] = random.randint(0, len(FUNCTIONS) - 1)
                elif gene_type in [1, 2]:  # Input connection genes
                    new_genome[i] = random.randint(
                        max(0, node_index - N_BACK),
                        node_index + 3 - 1
                    )
                else:  # Parameter gene
                    new_genome[i] = random.uniform(-1.0, 1.0)
            else:  # Output gene
                new_genome[i] = random.randint(3, N_NODES + 2)
                
    return new_genome


def evaluate(genome: List[float], image: np.ndarray) -> np.ndarray:
    """Evaluate a CGP individual on an input image"""
    # Initialize node outputs with input image channels
    node_outputs = [image[..., i] for i in range(image.shape[-1])]
    
    # Process each node
    for i in range(N_NODES):
        idx = i * GENES_PER_NODE
        func_id = int(genome[idx])
        input1_idx = int(genome[idx + 1])
        input2_idx = int(genome[idx + 2])
        param = genome[idx + 3]
        
        func = FUNCTIONS[func_id]
        
        if func.n_inputs == 1:
            output = func.func(node_outputs[input1_idx], param)
        else:
            output = func.func(node_outputs[input1_idx], 
                                node_outputs[input2_idx], param)
        
        node_outputs.append(output)
    
    # Collect output features
    features = []
    output_genes = genome[-N_OUTPUTS:]
    for output_idx in output_genes:
        output_idx = int(output_idx)
        feature = node_outputs[output_idx]
        # Global average pooling to get a single value per feature map
        features.append(np.mean(feature))
        
    return np.array(features)


def evolve(train_images: List[np.ndarray],
            train_labels: List[int],
            test_images: List[np.ndarray],
            test_labels: List[int],
            eval_model: Type,
            n_generations: int = 100,
            population_size: int = 50,
            mutation_rate: float = 0.1) -> Tuple[List[float], float]:
    """
    Evolve the CGP population
    
    Args:
        train_images: List of training images
        train_labels: List of training labels
        test_images: List of testing images
        test_labels: List of testing labels
        n_generations: Number of generations to evolve
        population_size: Size of the population
        mutation_rate: Probability of mutation per gene
        eval_model: The model define to evaluate the genoma secuence
        
    Returns:
        best_genome: Best performing genome
        best_fitness: Fitness of the best genome
    """
    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    
    # Convert labels to one-hot encoding
    n_classes = len(set(train_labels))
    labels_onehot = np.eye(n_classes)[train_labels]
    labels_test_onehot = np.eye(n_classes)[test_labels]
    
    def compute_fitness(genome: List[float]) -> float:
        """Compute fitness as classification accuracy"""
        features_list = []
        for image in train_images:
            features = evaluate(genome, image)
            features_list.append(features)
        features_array = np.stack(features_list)

        features_test_list = []
        for image in test_images:
            features = evaluate(genome, image)
            features_test_list.append(features)
        features_test_array = np.stack(features_test_list)
        
        # training Model
        eval_model.fit(features_array, labels_onehot)
        return [evaluation.roc_auc_score(eval_model.predict(features_array), labels_onehot),
               evaluation.roc_auc_score(eval_model.predict(features_test_array), labels_test_onehot)]
        

    for generation in range(n_generations):
        # Evaluate fitness for all individuals
        fitnesses = [compute_fitness(genome) for genome in population]
        
        # Select best individual
        best_idx = np.argmax(fitnesses, axis=1)[1]
        best_genome = population[best_idx]
        best_train_fitness = fitnesses[best_idx][0]
        best_test_fitness = fitnesses[best_idx][1]
        
        print(f"""Generation {generation + 1}/{n_generations}, Best Training Fitness: {best_train_fitness:.4f}
                 Best Testing Fitness: {best_test_fitness:.4f}""")
        
        # Create new population through mutation
        new_population = [best_genome]  # Keep best individual (elitism)
        while len(new_population) < population_size:
            offspring = mutate(best_genome, mutation_rate)
            new_population.append(offspring)
        
        population = new_population
        
    return best_genome, best_fitness

# def extract_features(image_path: str, cgp_genome: List[float], cgp_instance: ImageCGP) -> np.ndarray:
#     """Utility function to extract features from a new image using trained CGP"""
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (64, 64))  # Resize to standard size
#     image = image.astype(np.float32) / 255.0  # Normalize
    
#     features = cgp_instance.evaluate(cgp_genome, image)
#     return features

# Example usage
if __name__ == "__main__":
    from vision import vision_functions
    from classifications import random_forests
    # Initialize CGP
    INPUT_SHAPE = (64, 64, 3)  # RGB images
    FUNCTIONS = vision_functions()
    N_NODES = get_n_nodes()
    
    # Generate some dummy data
    n_samples = 100
    dummy_images = [np.random.rand(64, 64, 3) for _ in range(n_samples)]
    dummy_labels = [random.randint(0, 9) for _ in range(n_samples)]
    dummy_train = dummy_images[:-10]
    dummy_train_label = dummy_labels[:-10]
    dummy_test = dummy_images[10:]
    dummy_test_label = dummy_labels[10:]
    
    # Evolve CGP
    best_genome, best_fitness = evolve(dummy_train, dummy_train_label, dummy_test, dummy_test_label, n_generations=10,
                                       population_size=50, eval_model=random_forests.classification_model())
    
    print(f"Training completed. Best fitness: {best_fitness}")
    print(f"best genome: {best_genome}")
    
    # Extract features from a new image
    # features = extract_features("path/to/image.jpg", best_genome, cgp)