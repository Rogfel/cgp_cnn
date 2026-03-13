import random
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Type, Optional, Any, Dict
import json
import os
import logging

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None
try:
    from sklearn.base import clone as sklearn_clone
except ImportError:
    sklearn_clone = None

# Configurar logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


#INPUT_SHAPE: (height, width, channels)
INPUT_SHAPE=None
#N_COLUMNS: Number of columns in CGP grid
N_COLUMNS=40
#N_ROWS: Number of rows in CGP grid
N_ROWS=10
#N_BACK: Number of columns to look back for connections
N_BACK=25
#N_OUTPUTS: Number of features to extract
N_OUTPUTS=32
# Define available functions
FUNCTIONS = None
# Calculate genome length
GENES_PER_NODE=4  # function_id, input1, input2, parameter
# numbers of nodes: N_COLUMNS * N_ROWS
N_NODES=None


def get_n_nodes():
    return N_COLUMNS * N_ROWS


def _extract_pooling_features(feature_map: np.ndarray) -> np.ndarray:
    """
    Extrai múltiplas estatísticas de pooling de um feature map.
    
    Args:
        feature_map: Array 2D ou 3D representando um feature map
        
    Returns:
        Array com estatísticas extraídas
    """
    # Garantir 2D
    if len(feature_map.shape) == 3:
        feature_map = feature_map.reshape(-1, feature_map.shape[-1])
    
    # Flatten para cálculo de estatísticas
    flat = feature_map.flatten()
    
    features = []
    
    # Estatísticas básicas
    features.append(np.mean(flat))
    features.append(np.std(flat))
    features.append(np.max(flat))
    features.append(np.min(flat))
    
    # Percentis
    features.append(np.percentile(flat, 25))
    features.append(np.percentile(flat, 50))  # mediana
    features.append(np.percentile(flat, 75))
    
    # Energia e outras métricas
    features.append(np.sum(flat ** 2) / flat.size)  # energia
    
    # Range dinâmico
    features.append(np.max(flat) - np.min(flat))
    
    return np.array(features)


def get_genome_length():
    return get_n_nodes() * GENES_PER_NODE + N_OUTPUTS


def validate_config():
    """Valida se a configuração do CGP está completa"""
    errors = []
    
    if INPUT_SHAPE is None:
        errors.append("INPUT_SHAPE must be set")
    if FUNCTIONS is None or len(FUNCTIONS) == 0:
        errors.append("FUNCTIONS must be initialized")
    if N_NODES is None:
        errors.append("N_NODES must be calculated")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


def _ensure_channels_last(image: np.ndarray) -> np.ndarray:
    """
    Converte imagem de (C, H, W) para (H, W, C) se necessário.
    Aceita saída do PyTorch ToTensor() que usa formato channels-first.
    """
    if len(image.shape) != 3:
        return image
    # (C, H, W) -> (H, W, C)
    if INPUT_SHAPE is not None and image.shape[0] == INPUT_SHAPE[2] and image.shape[1] == INPUT_SHAPE[0] and image.shape[2] == INPUT_SHAPE[1]:
        return np.transpose(image, (1, 2, 0))
    return image


def validate_image(image: np.ndarray) -> bool:
    """Valida formato de imagem (aceita HWC ou CHW e valida dimensões)."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be numpy array, got {type(image)}")
    
    if len(image.shape) != 3:
        raise ValueError(f"Image must be 3D (H, W, C) or (C, H, W), got shape {image.shape}")
    
    if INPUT_SHAPE is None:
        return True
    
    h, w, c = INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]
    # (H, W, C)
    if image.shape == (h, w, c):
        return True
    # (C, H, W) - será convertido em evaluate()
    if image.shape == (c, h, w):
        return True
    
    raise ValueError(f"Image shape {image.shape} doesn't match INPUT_SHAPE {INPUT_SHAPE} (or (C,H,W) { (c, h, w) })")


def create_individual() -> List[float]:
    """Create a random CGP individual"""
    if FUNCTIONS is None or len(FUNCTIONS) == 0:
        raise ValueError("FUNCTIONS must be initialized before creating individuals")
    if N_NODES is None:
        raise ValueError("N_NODES must be calculated before creating individuals")
    
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
    """
    Evaluate a CGP individual on an input image.
    
    Args:
        genome: List of genes representing the CGP individual
        image: Input image as numpy array with shape (height, width, channels)
    
    Returns:
        Array of extracted features with shape (N_OUTPUTS,)
    
    Raises:
        ValueError: If configuration is invalid or genome is malformed
        IndexError: If genome indices are out of bounds
    """
    # Validação inicial
    validate_config()
    validate_image(image)
    
    # Garantir formato (H, W, C) - converter de (C, H, W) se vier do PyTorch
    image = _ensure_channels_last(image.copy())
    
    if len(genome) != get_genome_length():
        raise ValueError(f"Genome length {len(genome)} doesn't match expected {get_genome_length()}")
    
    # Initialize node outputs with input image channels
    node_outputs = [image[..., i] for i in range(image.shape[-1])]
    
    # Process each node
    for i in range(N_NODES):
        idx = i * GENES_PER_NODE
        
        # Validação de índices do genoma
        if idx + 3 >= len(genome):
            raise IndexError(f"Genome too short for node {i}")
        
        func_id = int(genome[idx])
        input1_idx = int(genome[idx + 1])
        input2_idx = int(genome[idx + 2])
        param = genome[idx + 3]
        
        # Validação de função
        if func_id < 0 or func_id >= len(FUNCTIONS):
            logger.warning(f"Invalid function ID {func_id} at node {i}, using function 0")
            func_id = 0
        
        func = FUNCTIONS[func_id]
        
        # Validação de índices de entrada
        if input1_idx < 0 or input1_idx >= len(node_outputs):
            logger.warning(f"Invalid input1 index {input1_idx} at node {i}, using index 0")
            input1_idx = 0
        
        if func.n_inputs == 2:
            if input2_idx < 0 or input2_idx >= len(node_outputs):
                logger.warning(f"Invalid input2 index {input2_idx} at node {i}, using index 0")
                input2_idx = 0
        
        # Executar função com tratamento de erros
        try:
            if func.n_inputs == 1:
                output = func.func(node_outputs[input1_idx], param)
            else:
                output = func.func(node_outputs[input1_idx],
                                 node_outputs[input2_idx], param)
        except Exception as e:
            logger.warning(f"Error in node {i}, function {func.name}: {e}. Using zeros.")
            output = np.zeros_like(node_outputs[0])
        
        node_outputs.append(output)
    
    # Collect output features - multi-feature pooling
    output_genes = genome[-N_OUTPUTS:]
    features = []
    for output_idx in output_genes:
        idx = int(output_idx)
        if idx < 0 or idx >= len(node_outputs):
            logger.warning(f"Invalid output index {idx}, using index 0")
            idx = 0
        output = node_outputs[idx]
        # Multi-feature pooling (9 features por output)
        pooling_features = _extract_pooling_features(output)
        features.extend(pooling_features)
        
    return np.array(features)


def save_best_genome(genome: List[float], train_fitness: float, val_fitness: float, 
                     dt_model=None, filename: str = "best_genome.json"):
    """
    Save the best genome, its fitness metrics, and the trained DT model to files.
    
    Args:
        genome: The genome to save
        train_fitness: Training fitness score
        val_fitness: Validation fitness score
        dt_model: Optional trained decision tree model
        filename: Output filename for genome JSON
    
    Raises:
        IOError: If file operations fail
    """
    import pickle
    
    try:
        # Criar diretório se não existir
        dirname = os.path.dirname(filename) if os.path.dirname(filename) else '.'
        os.makedirs(dirname, exist_ok=True)
        
        # Save genome data
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
        logger.info(f"Genome saved to: {filename}")
        
        # Save DT model if provided
        if dt_model is not None:
            model_filename = filename.replace('.json', '_dt_model.pkl')
            with open(model_filename, 'wb') as f:
                pickle.dump(dt_model, f)
            logger.info(f"DT model saved to: {model_filename}")
        
    except IOError as e:
        logger.error(f"Error saving genome: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving genome: {e}")
        raise


def save_evolution_history(history: dict, filename: str = "evolution_history.json"):
    """
    Save the complete evolution history to a JSON file.
    
    Args:
        history: Dictionary containing evolution history
        filename: Output filename
    
    Raises:
        IOError: If file operations fail
    """
    try:
        # Criar diretório se não existir
        dirname = os.path.dirname(filename) if os.path.dirname(filename) else '.'
        os.makedirs(dirname, exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(history, f, indent=2)
        logger.debug(f"Evolution history saved to: {filename}")
    except IOError as e:
        logger.error(f"Error saving evolution history: {e}")
        raise


def _genome_cache_key(genome: List[float]) -> tuple:
    """Chave hashável para cache de fitness (tuple de floats)."""
    return tuple(genome)


def _compute_fitness_single(
    genome: List[float],
    train_images: List[np.ndarray],
    train_labels: List[int],
    eval_model: Any,
    use_tqdm: bool = True,
) -> Tuple[float, Any]:
    """
    Avalia um único genoma: extrai features, treina um clone do modelo, retorna (fitness, model).
    Usado com ou sem paralelização. Cada chamada usa um clone do eval_model.
    """
    if sklearn_clone is None:
        raise ImportError("sklearn.base.clone is required; install scikit-learn")
    features_array = np.stack([
        evaluate(genome, image)
        for image in (tqdm(train_images, desc="    Training Evaluation") if use_tqdm else train_images)
    ])
    
    # Tratar valores infinitos ou muito grandes
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=1e6, neginf=-1e6)
    features_array = np.clip(features_array, -1e6, 1e6)
    
    model = sklearn_clone(eval_model)
    model.fit(features_array, train_labels)
    fitness = model.score(features_array, train_labels)
    return (float(fitness), model)


def evolve(
    train_images: List[np.ndarray],
    train_labels: List[int],
    val_images: List[np.ndarray],
    val_labels: List[int],
    eval_model: Any,
    n_generations: int = 100,
    population_size: int = 50,
    mutation_rate: float = 0.1,
    n_jobs: int = 1,
    early_stopping_patience: Optional[int] = None,
) -> Tuple[List[float], float, float]:
    """
    Evolve the CGP population.

    Args:
        train_images: List of training images
        train_labels: List of training labels
        val_images: List of validation images
        val_labels: List of validation labels
        eval_model: Model instance (will be cloned per individual). Must be sklearn-like (fit, score).
        n_generations: Number of generations to evolve
        population_size: Size of the population
        mutation_rate: Probability of mutation per gene
        n_jobs: Number of parallel jobs for fitness evaluation. 1 = sequential, -1 = all cores.
        early_stopping_patience: Stop if validation fitness does not improve for this many generations.
            None = disabled.

    Returns:
        best_genome: Best performing genome
        best_fitness: Training fitness of the best genome
        val_fitness: Validation fitness of the best genome
    """
    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    best_fitness_overall = 0.0
    best_genome_overall = None
    best_val_fitness = 0.0
    best_model_overall = None

    # Cache de fitness: evita re-avaliar o mesmo genoma (key = tuple(genome) -> (fitness, model))
    fitness_cache: Dict[tuple, Tuple[float, Any]] = {}

    # Early stopping
    generations_without_improvement = 0
    if early_stopping_patience is not None and early_stopping_patience < 1:
        early_stopping_patience = None

    # Initialize evolution history
    evolution_history = {
        "generations": [],
        "best_training_fitness": [],
        "best_validation_fitness": [],
        "population_training_fitness": [],
        "population_validation_fitness": [],
        "best_genome": None,
        "final_best_training_fitness": 0,
        "final_best_validation_fitness": 0,
    }

    def compute_validation_fitness(genome: List[float], model: Any) -> float:
        """Calcula fitness de validação usando o modelo já treinado."""
        features_array = np.stack([
            evaluate(genome, image)
            for image in tqdm(val_images, desc="    Validation Evaluation")
        ])
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1e6, neginf=-1e6)
        features_array = np.clip(features_array, -1e6, 1e6)
        return float(model.score(features_array, val_labels))

    for generation in range(n_generations):
        # --- Avaliar fitness da população (com cache e opcionalmente em paralelo) ---
        results: List[Optional[Tuple[float, Any]]] = [None] * len(population)
        to_compute: List[Tuple[int, List[float]]] = []

        for i, genome in enumerate(population):
            key = _genome_cache_key(genome)
            if key in fitness_cache:
                results[i] = fitness_cache[key]
            else:
                to_compute.append((i, genome))

        if to_compute:
            if n_jobs == 1 or Parallel is None or delayed is None:
                iterator = tqdm(to_compute, desc="    Individuals") if len(to_compute) > 1 else to_compute
                for i, genome in iterator:
                    fit, model = _compute_fitness_single(
                        genome, train_images, train_labels, eval_model, use_tqdm=(len(to_compute) == 1)
                    )
                    results[i] = (fit, model)
                    fitness_cache[_genome_cache_key(genome)] = (fit, model)
            else:
                computed = Parallel(n_jobs=n_jobs)(
                    delayed(_compute_fitness_single)(
                        genome, train_images, train_labels, eval_model, use_tqdm=False
                    )
                    for _, genome in to_compute
                )
                for idx, (i, genome) in enumerate(to_compute):
                    fit, model = computed[idx]
                    results[i] = (fit, model)
                    fitness_cache[_genome_cache_key(genome)] = (fit, model)

        fitnesses = [r[0] for r in results]
        trained_models = [r[1] for r in results]

        # Select best individual
        best_idx = int(np.argmax(fitnesses))
        current_best_genome = population[best_idx]
        current_best_fitness = fitnesses[best_idx]
        current_best_model = trained_models[best_idx]

        # Validation fitness do melhor
        current_val_fitness = compute_validation_fitness(current_best_genome, current_best_model)

        # Evolution history
        evolution_history["generations"].append(generation + 1)
        evolution_history["best_training_fitness"].append(float(current_best_fitness))
        evolution_history["best_validation_fitness"].append(float(current_val_fitness))
        evolution_history["population_training_fitness"].append([float(f) for f in fitnesses])

        # Atualizar melhor global e early stopping
        if current_val_fitness > best_val_fitness:
            best_genome_overall = current_best_genome
            best_fitness_overall = current_best_fitness
            best_val_fitness = current_val_fitness
            best_model_overall = current_best_model
            generations_without_improvement = 0
            save_best_genome(
                best_genome_overall, best_fitness_overall, best_val_fitness, best_model_overall
            )
            save_evolution_history(evolution_history)
        else:
            generations_without_improvement += 1

        logger.info(
            "Generation %d/%d | Train fitness: %.4f | Val fitness: %.4f | Cache size: %d",
            generation + 1,
            n_generations,
            current_best_fitness,
            current_val_fitness,
            len(fitness_cache),
        )

        if (
            early_stopping_patience is not None
            and generations_without_improvement >= early_stopping_patience
        ):
            logger.info(
                "Early stopping at generation %d (no improvement for %d generations)",
                generation + 1,
                early_stopping_patience,
            )
            break

        # Nova população (elitismo + torneio)
        new_population = [current_best_genome]
        tournament_size = 3
        pop_fitness = list(zip(population, fitnesses))
        while len(new_population) < population_size:
            if random.random() < 0.7:
                parent1 = max(random.sample(pop_fitness, tournament_size), key=lambda x: x[1])[0]
                parent2 = max(random.sample(pop_fitness, tournament_size), key=lambda x: x[1])[0]
                child = mutate(crossover(parent1, parent2), mutation_rate)
            else:
                parent = max(random.sample(pop_fitness, tournament_size), key=lambda x: x[1])[0]
                child = mutate(parent, mutation_rate)
            new_population.append(child)
        population = new_population

    evolution_history["final_best_training_fitness"] = float(best_fitness_overall or 0)
    evolution_history["final_best_validation_fitness"] = float(best_val_fitness or 0)
    save_evolution_history(evolution_history)

    return best_genome_overall, best_fitness_overall, best_val_fitness


def load_best_genome_and_model(genome_filename: str = "best_genome.json"):
    """
    Load the best genome and its corresponding DT model.
    
    Args:
        genome_filename: Path to the genome JSON file
    
    Returns:
        Tuple of (genome_data dict, dt_model or None)
    
    Raises:
        FileNotFoundError: If genome file doesn't exist
        IOError: If file operations fail
    """
    import pickle
    
    try:
        # Load genome data
        if not os.path.exists(genome_filename):
            raise FileNotFoundError(f"Genome file not found: {genome_filename}")
        
        with open(genome_filename, 'r') as f:
            genome_data = json.load(f)
        
        logger.info(f"Loaded genome from: {genome_filename}")
        
        # Load DT model
        model_filename = genome_filename.replace('.json', '_dt_model.pkl')
        try:
            if not os.path.exists(model_filename):
                logger.warning(f"DT model file not found: {model_filename}")
                return genome_data, None
            
            with open(model_filename, 'rb') as f:
                dt_model = pickle.load(f)
            logger.info(f"Loaded DT model from: {model_filename}")
            return genome_data, dt_model
        except FileNotFoundError:
            logger.warning(f"DT model file not found: {model_filename}")
            return genome_data, None
        except Exception as e:
            logger.error(f"Error loading DT model: {e}")
            return genome_data, None
            
    except IOError as e:
        logger.error(f"Error loading genome: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing genome JSON: {e}")
        raise


def predict_with_saved_model(genome_data: dict, dt_model, image: np.ndarray):
    """Make prediction using saved genome and DT model"""
    if dt_model is None:
        raise ValueError("DT model is None. Cannot make predictions.")
    
    # Extract genome
    genome = genome_data['genome']
    
    # Evaluate genome to get features
    features = evaluate(genome, image)
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    features = np.clip(features, -1e6, 1e6)
    
    # Make prediction
    prediction = dt_model.predict([features])
    prediction_proba = dt_model.predict_proba([features])
    
    return prediction[0], prediction_proba[0]


# ==================== APRENDIZADO ATIVO ====================

def compute_uncertainty_scores(
    genome: List[float],
    unlabeled_images: List[np.ndarray],
    classifier: Any
) -> np.ndarray:
    """
    Calcula scores de incerteza para imagens não rotuladas.
    Usa a entropia das probabilidades previstas.
    
    Args:
        genome: Genoma do CGP
        unlabeled_images: Lista de imagens não rotuladas
        classifier: Classificador sklearn-like
    
    Returns:
        Array de scores de incerteza (maior = mais incerto)
    """
    uncertainties = []
    
    for image in unlabeled_images:
        features = evaluate(genome, image)
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        features = np.clip(features, -1e6, 1e6)
        features_2d = features.reshape(1, -1)
        
        # Usar modelo se disponível, senão skip
        if hasattr(classifier, 'predict_proba'):
            proba = classifier.predict_proba(features_2d)
            # Entropia: -sum(p * log(p))
            proba = np.clip(proba, 1e-10, 1)  # Evitar log(0)
            entropy = -np.sum(proba * np.log(proba))
            uncertainties.append(entropy)
        elif hasattr(classifier, 'predict'):
            # Se só tem predict, usar distância da decisão
            pred = classifier.predict(features_2d)
            uncertainties.append(0.0)  # Placeholder
        else:
            uncertainties.append(0.0)
    
    return np.array(uncertainties)


def select_uncertain_samples(
    genome: List[float],
    unlabeled_images: List[np.ndarray],
    classifier: Any,
    n_samples: int = 5
) -> Tuple[List[int], np.ndarray]:
    """
    Seleciona as n_samples mais incertas do pool não rotulado.
    
    Args:
        genome: Genoma do CGP
        unlabeled_images: Pool de imagens não rotuladas
        classifier: Classificador sklearn-like
        n_samples: Número de amostras a selecionar
    
    Returns:
        Tuple de (índices selecionados, scores de incerteza)
    """
    if len(unlabeled_images) <= n_samples:
        return list(range(len(unlabeled_images))), np.arange(len(unlabeled_images))
    
    uncertainties = compute_uncertainty_scores(genome, unlabeled_images, classifier)
    
    # Selecionar as mais incertas (maior entropia)
    indices = np.argsort(uncertainties)[-n_samples:]
    
    return indices.tolist(), uncertainties


def query_by_committee(
    population: List[List[float]],
    unlabeled_images: List[np.ndarray],
    n_samples: int = 5
) -> Tuple[List[int], np.ndarray]:
    """
    Query-by-Committee: seleciona amostras onde o committee discorda mais.
    
    Args:
        population: Lista de genomas (committee)
        unlabeled_images: Pool de imagens não rotuladas
        n_samples: Número de amostras a selecionar
    
    Returns:
        Tuple de (índices selecionados, scores de discordância)
    """
    disagreements = []
    
    for image in unlabeled_images:
        features_list = [evaluate(genome, image) for genome in population]
        
        # Calcular variância das features (proxy de discordância)
        features_array = np.stack(features_list)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1e6, neginf=-1e6)
        features_array = np.clip(features_array, -1e6, 1e6)
        variance = np.var(features_array, axis=0)
        disagreement = np.mean(variance)
        disagreements.append(disagreement)
    
    disagreements = np.array(disagreements)
    
    if len(unlabeled_images) <= n_samples:
        return list(range(len(unlabeled_images))), disagreements
    
    indices = np.argsort(disagreements)[-n_samples:]
    
    return indices.tolist(), disagreements


def evolve_active_learning(
    train_images: List[np.ndarray],
    train_labels: List[int],
    val_images: List[np.ndarray],
    val_labels: List[int],
    unlabeled_images: List[np.ndarray],  # Pool de amostras não rotuladas
    eval_model: Any,
    n_al_iterations: int = 10,
    samples_per_iteration: int = 5,
    n_generations_per_al: int = 50,
    population_size: int = 50,
    mutation_rate: float = 0.1,
    early_stopping_patience: Optional[int] = 50,
    query_strategy: str = "uncertainty",  # "uncertainty" ou "committee"
    save_models_dir: str = "saved_models_al",
) -> Tuple[List[float], float, float, dict]:
    """
    Evolui CGP com Aprendizado Ativo.
    
    Args:
        train_images: Imagens de treino iniciais
        train_labels: Labels de treino iniciais
        val_images: Imagens de validação
        val_labels: Labels de validação
        unlabeled_images: Pool de imagens não rotuladas
        eval_model: Modelo sklearn-like
        n_al_iterations: Número de iterações de aprendizado ativo
        samples_per_iteration: Amostras a adicionar por iteração
        n_generations_per_al: Gerações de evolução por iteração AL
        population_size: Tamanho da população CGP
        mutation_rate: Taxa de mutação
        early_stopping_patience: Paciência para early stopping
        query_strategy: "uncertainty" (incerteza) ou "committee" (QBC)
        save_models_dir: Diretório para salvar os melhores modelos de cada iteração
    
    Returns:
        Tuple de (best_genome, best_train_fitness, best_val_fitness, history)
    """
    # Criar diretório para salvar modelos se não existir
    os.makedirs(save_models_dir, exist_ok=True)
    # Copiar pools para não modificar originais
    current_train_images = list(train_images)
    current_train_labels = list(train_labels)
    current_unlabeled = list(unlabeled_images)
    
    # Histórico do aprendizado ativo
    al_history = {
        "iterations": [],
        "n_train_samples": [],
        "best_fitness": [],
        "val_fitness": [],
        "selected_indices": []
    }
    
    logger.info(f"=== Aprendizado Ativo: {n_al_iterations} iterações ===")
    logger.info(f"Amostras iniciais: {len(current_train_images)}")
    logger.info(f"Pool não rotulado: {len(current_unlabeled)}")
    
    for al_iter in range(n_al_iterations):
        logger.info(f"\n--- AL Iteração {al_iter + 1}/{n_al_iterations} ---")
        logger.info(f"Conjunto de treino: {len(current_train_images)} amostras")
        
        # 1. Evoluir CGP com o conjunto atual
        best_genome, best_fitness, val_fitness = evolve(
            train_images=current_train_images,
            train_labels=current_train_labels,
            val_images=val_images,
            val_labels=val_labels,
            eval_model=eval_model,
            n_generations=n_generations_per_al,
            population_size=population_size,
            mutation_rate=mutation_rate,
            n_jobs=1,
            early_stopping_patience=early_stopping_patience,
        )
        
        # 2. Treinar classificador final para pseudo-labels
        final_model = sklearn_clone(eval_model)
        features_array = np.stack([evaluate(best_genome, img) for img in current_train_images])
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1e6, neginf=-1e6)
        features_array = np.clip(features_array, -1e6, 1e6)
        final_model.fit(features_array, current_train_labels)
        
        # 3. Selecionar novas amostras
        if query_strategy == "uncertainty":
            selected_indices, scores = select_uncertain_samples(
                best_genome, current_unlabeled, final_model, samples_per_iteration
            )
        else:  # committee
            # Usar população como committee
            population = [create_individual() for _ in range(min(10, population_size))]
            selected_indices, scores = query_by_committee(
                population, current_unlabeled, samples_per_iteration
            )
        
        # 4. Adicionar amostras selecionadas ao treino
        newly_labeled = [current_unlabeled[i] for i in selected_indices]
        
        # Gerar pseudo-labels
        new_features = np.stack([evaluate(best_genome, img) for img in newly_labeled])
        new_features = np.nan_to_num(new_features, nan=0.0, posinf=1e6, neginf=-1e6)
        new_features = np.clip(new_features, -1e6, 1e6)
        pseudo_labels = final_model.predict(new_features)
        
        current_train_images.extend(newly_labeled)
        current_train_labels.extend(pseudo_labels.tolist())
        
        # 5. Remover do pool não rotulado
        current_unlabeled = [img for i, img in enumerate(current_unlabeled) 
                            if i not in selected_indices]
        
        # Registrar histórico
        al_history["iterations"].append(al_iter + 1)
        al_history["n_train_samples"].append(len(current_train_images))
        al_history["best_fitness"].append(float(best_fitness))
        al_history["val_fitness"].append(float(val_fitness))
        al_history["selected_indices"].append(selected_indices)
        
        logger.info(f"Adicionadas {len(newly_labeled)} amostras")
        logger.info(f"Fitness treino: {best_fitness:.4f}, Val: {val_fitness:.4f}")
        logger.info(f"Pool restante: {len(current_unlabeled)} amostras")
        
        # Salvar o melhor modelo desta iteração
        iteration_filename = os.path.join(save_models_dir, f"best_genome_iter_{al_iter + 1:03d}.json")
        save_best_genome(
            genome=best_genome,
            train_fitness=best_fitness,
            val_fitness=val_fitness,
            dt_model=final_model,
            filename=iteration_filename
        )
        logger.info(f"Modelo da iteracao {al_iter + 1} salvo em: {iteration_filename}")
        
        # Early stopping: se pool acabou
        if len(current_unlabeled) < samples_per_iteration:
            logger.info("Pool de amostras não rotuladas esgotado!")
            break
    
    # Evolução final com todo o conjunto rotulado
    logger.info("\n=== Evolução Final com Todas as Amostras ===")
    best_genome, best_fitness, val_fitness = evolve(
        train_images=current_train_images,
        train_labels=current_train_labels,
        val_images=val_images,
        val_labels=val_labels,
        eval_model=eval_model,
        n_generations=n_generations_per_al * 2,
        population_size=population_size,
        mutation_rate=mutation_rate,
        early_stopping_patience=None,
    )
    
    al_history["final_genome"] = best_genome
    al_history["final_train_fitness"] = float(best_fitness)
    al_history["final_val_fitness"] = float(val_fitness)
    
    # Salvar o melhor modelo final
    final_filename = os.path.join(save_models_dir, "best_genome_final.json")
    save_best_genome(
        genome=best_genome,
        train_fitness=best_fitness,
        val_fitness=val_fitness,
        dt_model=None,
        filename=final_filename
    )
    logger.info(f"Modelo final salvo em: {final_filename}")
    logger.info(f"Todos os modelos salvos em: {save_models_dir}")
    
    return best_genome, best_fitness, val_fitness, al_history


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