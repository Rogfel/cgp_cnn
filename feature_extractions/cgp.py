import random
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Type
import json


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
# numbers of nodes: N_COLUMNS * N_ROWS
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
        for _ in range(2):  # Always store 2 inputs, even if not used
            x = random.randint(max(0, i - N_BACK), i + 3 - 1)  # +3 for RGB channels
            genome.append(x)
        
        # Parameter gene
        genome.append(random.uniform(-1.0, 1.0))
    
    # Output connection genes
    for _ in range(N_OUTPUTS):
        genome.append(random.randint(3, N_NODES + 2))  # +3 for RGB channels, -1 for 0-based
        
    return genome


def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Perform crossover between two parents"""
    child = []
    # Single point crossover
    crossover_point = random.randint(0, len(parent1) - 1)
    child.extend(parent1[:crossover_point])
    child.extend(parent2[crossover_point:])
    return child


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
    output_genes = genome[-N_OUTPUTS:]
    features = []
    for output_idx in output_genes:
        output = node_outputs[int(output_idx)]
        # Global average pooling for each feature map
        features.append(np.mean(output))
        
    return np.array(features)


def save_best_genome(genome: List[float], train_fitness: float, val_fitness: float, filename: str = "best_genome.json"):
    """Save the best genome and its fitness metrics to a JSON file"""
    data = {
        "genome": genome,
        "training_fitness": train_fitness,
        "validation_fitness": val_fitness,
        "hyperparameters": {
            "n_columns": N_COLUMNS,
            "n_rows": N_ROWS,
            "n_back": N_BACK,
            "n_outputs": N_OUTPUTS,
            "genes_per_node": GENES_PER_NODE
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def evolve(train_images: List[np.ndarray],
          train_labels: List[int],
          val_images: List[np.ndarray],
          val_labels: List[int],
          eval_model: Type,
          n_generations: int = 100,
          population_size: int = 50,
          mutation_rate: float = 0.1) -> Tuple[List[float], float, float]:
    """
    Evolve the CGP population
    
    Args:
        train_images: List of training images
        train_labels: List of training labels
        val_images: List of validation images
        val_labels: List of validation labels
        eval_model: The model to evaluate the genome sequence
        n_generations: Number of generations to evolve
        population_size: Size of the population
        mutation_rate: Probability of mutation per gene
        
    Returns:
        best_genome: Best performing genome
        best_fitness: Training fitness of the best genome
        val_fitness: Validation fitness of the best genome
    """
    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    best_fitness_overall = 0
    best_genome_overall = None
    best_val_fitness = 0
    
    def compute_fitness(genome: List[float], images: List[np.ndarray], labels: List[int], is_training: bool = True) -> float:
        """Compute fitness as classification accuracy"""
        features_array = np.stack([evaluate(genome, image) for image in tqdm(images, 
            desc="    Training Evaluation" if is_training else "    Validation Evaluation")])
        
        if is_training:
            # Fit model on training data
            eval_model.fit(features_array, labels)
            return eval_model.score(features_array, labels)
        else:
            # Evaluate on validation data
            return eval_model.score(features_array, labels)

    for generation in range(n_generations):
        # Evaluate fitness for all individuals
        fitnesses = [compute_fitness(genome, train_images, train_labels) for genome in population]
        
        # Select best individual
        best_idx = np.argmax(fitnesses)
        current_best_genome = population[best_idx]
        current_best_fitness = fitnesses[best_idx]
        
        # Compute validation fitness for best individual
        current_val_fitness = compute_fitness(current_best_genome, val_images, val_labels, is_training=False)
        
        # Update overall best if validation fitness improves
        if current_val_fitness > best_val_fitness:
            best_genome_overall = current_best_genome
            best_fitness_overall = current_best_fitness
            best_val_fitness = current_val_fitness
            # Save the best genome whenever we find a better one
            save_best_genome(best_genome_overall, best_fitness_overall, best_val_fitness)
        
        print(f"Generation {generation + 1}/{n_generations}")
        print(f"Best Training Fitness: {current_best_fitness:.4f}")
        print(f"Validation Fitness: {current_val_fitness:.4f}")
        
        # Create new population
        new_population = [current_best_genome]  # Elitism
        
        # Tournament selection
        tournament_size = 3
        while len(new_population) < population_size:
            if random.random() < 0.7:  # 70% chance of crossover
                # Select parents through tournament selection
                parent1 = max(random.sample(list(zip(population, fitnesses)), tournament_size), 
                            key=lambda x: x[1])[0]
                parent2 = max(random.sample(list(zip(population, fitnesses)), tournament_size), 
                            key=lambda x: x[1])[0]
                child = crossover(parent1, parent2)
                child = mutate(child, mutation_rate)
            else:
                # Select parent through tournament selection
                parent = max(random.sample(list(zip(population, fitnesses)), tournament_size), 
                           key=lambda x: x[1])[0]
                child = mutate(parent, mutation_rate)
            
            new_population.append(child)
        
        population = new_population
        
    return best_genome_overall, best_fitness_overall, best_val_fitness


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
    
    # Split into train and validation
    train_size = int(0.8 * n_samples)
    dummy_train = dummy_images[:train_size]
    dummy_train_label = dummy_labels[:train_size]
    dummy_val = dummy_images[train_size:]
    dummy_val_label = dummy_labels[train_size:]
    
    # Evolve CGP
    best_genome, best_fitness, val_fitness = evolve(
        dummy_train, dummy_train_label,
        dummy_val, dummy_val_label,
        n_generations=10,
        population_size=50,
        eval_model=random_forests.classification_model()
    )
    
    print(f"Training completed.")
    print(f"Best training fitness: {best_fitness:.4f}")
    print(f"Validation fitness: {val_fitness:.4f}")
    print(f"Best genome: {best_genome}")