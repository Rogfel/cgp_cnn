import random
import warnings
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Type
from classifications import evaluation
from feature_extractions import vision2 as vision


warnings.filterwarnings('ignore')

# Global variables
INPUT_SHAPE = None
N_COLUMNS = 10
N_ROWS = 1
N_BACK = 5
N_OUTPUTS = 0 # 64
FUNCTIONS = None
GENES_PER_NODE = 3
N_NODES = None

def get_n_nodes():
    """Get number of nodes"""
    return N_COLUMNS * N_ROWS

def get_genome_length():
    """Get genome length"""
    return get_n_nodes() * GENES_PER_NODE + N_OUTPUTS

class CGPEvaluator:
    """Class to handle CGP evaluation"""
    def __init__(self):
        self.feature_cache = {}
    
    @staticmethod
    def evaluate_image(genome: List[float], image: np.ndarray) -> np.ndarray:
        """Evaluate single image with error handling"""
        try:
            # Initialize node outputs with input image channels
            node_outputs = [vision.reshape(image)]
            
            # Process each node
            for i in range(N_NODES):
                try:
                    idx = i * GENES_PER_NODE           
                    func = FUNCTIONS[genome[idx]]
                    
                    if func.n_inputs == 1:
                        output = func.func(node_outputs[genome[idx + 1]])
                    else:
                        output = func.func(node_outputs[genome[idx + 1]], node_outputs[genome[idx + 2]])                     
                    
                    # Handle NaN and Inf values
                    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
                    node_outputs.append(output)

                except Exception as e:
                    print(f"Node processing error: {e}")
                    node_outputs.append(np.zeros_like(node_outputs[0]))
            
            last_output = vision.flatten(node_outputs[-1])
            return np.array(last_output[0], dtype=np.float32)
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return np.zeros(N_OUTPUTS, dtype=np.float32)

def create_individual() -> List[float]:
    """Create a random CGP individual with bounds checking"""
    genome = []
    
    # Generate nodes
    for i in range(N_NODES):
        # Function gene
        genome += [random.randint(0, len(FUNCTIONS) - 1)]
        
        # Input connection genes
        genome += [random.randint(0, i)] # connection 1
        genome += [random.randint(0, i)] # connection 2
        while genome[-2] == genome[-1] and i != 0:
            genome[-1] = random.randint(0, i)
        # # Parameter gene
        # genome.append(random.uniform(-1.0, 1.0))
    return genome

def mutate(genome: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Mutate a CGP individual with bounds checking"""
    new_genome = genome.copy()
    
    for enum, idx in enumerate(range(0, len(new_genome), GENES_PER_NODE)):
        if random.random() < mutation_rate:

            new_genome[idx] = random.randint(0, len(FUNCTIONS) - 1)
            new_genome[idx + 1] = random.randint(0, enum) # connection 1
            new_genome[idx + 2] = random.randint(0, enum) # connection 2
            while new_genome[idx + 1] == new_genome[idx + 2] and idx != 0:
                new_genome[idx + 2] = random.randint(0, enum)
            # else:  # Parameter gene
            #     new_genome[i] = random.uniform(-1.0, 1.0)    
    return new_genome

def compute_fitness(genome: List[float],
                    train_images: List[np.ndarray],
                    train_labels: np.ndarray,
                    eval_model: Type) -> float:
    """Worker function for parallel fitness computation"""
    
    try:
        evaluator = CGPEvaluator()
        
        # Extract features
        features_train = np.array([evaluator.evaluate_image(genome, img)
                                   for img in tqdm(train_images, desc="    Genoma Evaluation Progress")])

        # Ensure features are finite
        features_train = np.nan_to_num(features_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Fit model and compute scores
        eval_model.fit(features_train, train_labels)
            
        return eval_model.score(features_train, train_labels)
        
    except Exception as e:
        print(f"Fitness computation error: {e}")
        return 0.0

def evolve(train_images: List[np.ndarray],
          train_labels: List[int],
          eval_model: Type,
          n_generations: int = 100,
          population_size: int = 50,
          mutation_rate: float = 0.1) -> Tuple[List[float], float]:
    """Evolve the CGP population with error handling and parallel processing"""
    
    try:
        # Initialize population
        population = [create_individual() for _ in range(population_size)]

        # Convert labels to one-hot encoding
        n_classes = len(set(train_labels))
        labels_onehot = np.eye(n_classes)[train_labels]
                
        best_genome = None
        best_fitness = float('-inf')
        
        # Progress bar for generations
        for generation in range(n_generations):
            try:
                print(f"\nGeneration {generation + 1}/{n_generations}")
                # Prepare arguments for processing
                fitnesses = [compute_fitness(genome, train_images, labels_onehot, eval_model)
                        for genome in population]                
            
                # Convert to numpy array for faster operations
                # Find best individual
                best_idx = np.argmax(np.array(fitnesses))
                current_best_genome = population[best_idx]
                current_best_fitness = fitnesses[best_idx]
                
                # Update best overall
                if current_best_fitness > best_fitness:
                    best_genome = current_best_genome.copy()
                    best_fitness = current_best_fitness
                
                print(f"Best Fitness: {fitnesses[best_idx]:.4f}")
                
                # Create new elite population                
                # Fill rest of population with mutated versions of best
                population = [mutate(best_genome, mutation_rate)
                              for _ in range(population_size - 1)]
                population.append(best_genome)
                
            except Exception as e:
                print(f"Generation error: {e}")
                continue
        
        return best_genome, best_fitness
        
    except Exception as e:
        print(f"Evolution error: {e}")
        return create_individual(), float('-inf')

if __name__ == "__main__":
    from vision2 import vision_functions
    from classifications import random_forests
    
    try:
        # Initialize CGP
        INPUT_SHAPE = (64, 64, 3)
        FUNCTIONS = vision_functions()
        N_NODES = get_n_nodes()
        
        # Generate test data
        n_samples = 10
        dummy_images = [np.random.rand(64, 64, 3) for _ in range(n_samples)]
        dummy_labels = [random.randint(0, 1) for _ in range(n_samples)]  # Binary classification for testing
        
        # Create evaluation model
        model = random_forests.classification_model()
        
        # Run evolution
        best_genome, best_fitness = evolve(
            train_images=dummy_images,
            train_labels=dummy_labels,
            eval_model=model,
            n_generations=10,
            population_size=5,
            mutation_rate=0.1
        )
        
        print(f"\nTraining completed.")
        print(f"Best fitness: {best_fitness:.4f}")
        print(f"Genome length: {len(best_genome)}")
        
    except Exception as e:
        print(f"Main execution error: {e}")