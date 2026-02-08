# Revisão de Código - CGP CNN

## 📋 Resumo Executivo

Este documento apresenta uma análise detalhada do código e sugestões de melhorias organizadas por categoria de impacto.

---

## 🔴 Problemas Críticos

### 1. **Falta de Validação de Entrada no `evaluate()`**

**Problema:** A função `evaluate()` não valida índices de entrada, podendo causar `IndexError`.

**Localização:** `feature_extractions/cgp.py:94-125`

**Risco:** Alto - Pode causar crash durante a evolução

**Solução:**
```python
def evaluate(genome: List[float], image: np.ndarray) -> np.ndarray:
    """Evaluate a CGP individual on an input image"""
    # Validação inicial
    if FUNCTIONS is None:
        raise ValueError("FUNCTIONS must be initialized before evaluation")
    if N_NODES is None:
        raise ValueError("N_NODES must be initialized before evaluation")
    
    # Initialize node outputs with input image channels
    node_outputs = [image[..., i] for i in range(image.shape[-1])]
    
    # Process each node
    for i in range(N_NODES):
        idx = i * GENES_PER_NODE
        func_id = int(genome[idx])
        input1_idx = int(genome[idx + 1])
        input2_idx = int(genome[idx + 2])
        param = genome[idx + 3]
        
        # Validação de índices
        if func_id < 0 or func_id >= len(FUNCTIONS):
            raise ValueError(f"Invalid function ID: {func_id}")
        
        func = FUNCTIONS[func_id]
        
        # Validação de índices de entrada
        if input1_idx < 0 or input1_idx >= len(node_outputs):
            raise ValueError(f"Invalid input1 index: {input1_idx}")
        if func.n_inputs == 2:
            if input2_idx < 0 or input2_idx >= len(node_outputs):
                raise ValueError(f"Invalid input2 index: {input2_idx}")
        
        try:
            if func.n_inputs == 1:
                output = func.func(node_outputs[input1_idx], param)
            else:
                output = func.func(node_outputs[input1_idx],
                                 node_outputs[input2_idx], param)
        except Exception as e:
            # Log erro e retornar zeros para evitar crash
            print(f"Warning: Error in node {i}, function {func.name}: {e}")
            output = np.zeros_like(node_outputs[0])
        
        node_outputs.append(output)
    
    # Collect output features
    output_genes = genome[-N_OUTPUTS:]
    features = []
    for output_idx in output_genes:
        idx = int(output_idx)
        if idx < 0 or idx >= len(node_outputs):
            raise ValueError(f"Invalid output index: {idx}")
        output = node_outputs[idx]
        # Global average pooling for each feature map
        features.append(np.mean(output))
        
    return np.array(features)
```

### 2. **Problema de Performance: Re-treinamento do Modelo a Cada Geração**

**Problema:** Na função `evolve()`, o modelo é re-treinado para cada indivíduo da população a cada geração, mesmo quando o genoma não mudou.

**Localização:** `feature_extractions/cgp.py:211-222`

**Impacto:** Muito alto - Reduz drasticamente a velocidade de evolução

**Solução:** Cache de fitness e modelos treinados
```python
def evolve(...):
    # Adicionar cache
    fitness_cache = {}
    model_cache = {}
    
    def compute_fitness(genome: List[float], images: List[np.ndarray], 
                       labels: List[int], is_training: bool = True):
        # Criar hash do genoma para cache
        genome_hash = hash(tuple(genome))
        
        if is_training and genome_hash in fitness_cache:
            return fitness_cache[genome_hash], model_cache[genome_hash]
        
        features_array = np.stack([evaluate(genome, image) for image in tqdm(images, 
            desc="    Training Evaluation" if is_training else "    Validation Evaluation")])
        
        if is_training:
            # Criar nova instância do modelo para evitar reuso
            model = type(eval_model)()  # Nova instância
            model.fit(features_array, labels)
            fitness = model.score(features_array, labels)
            
            # Cache
            fitness_cache[genome_hash] = fitness
            model_cache[genome_hash] = model
            
            return fitness, model
        else:
            return eval_model.score(features_array, labels)
```

### 3. **Problema de Memória: Acúmulo de Modelos Treinados**

**Problema:** Todos os modelos treinados são mantidos em memória na lista `trained_models`.

**Localização:** `feature_extractions/cgp.py:228`

**Solução:** Manter apenas o melhor modelo
```python
# Em vez de:
trained_models = [result[1] for result in fitness_results]

# Usar:
best_model = trained_models[best_idx]  # Apenas o melhor
```

---

## 🟡 Problemas Importantes

### 4. **Falta de Tratamento de Erros em Operações de I/O**

**Problema:** Funções `save_best_genome()` e `load_best_genome_and_model()` não tratam erros adequadamente.

**Localização:** `feature_extractions/cgp.py:128-312`

**Solução:**
```python
def save_best_genome(genome: List[float], train_fitness: float, 
                     val_fitness: float, dt_model=None, 
                     filename: str = "best_genome.json"):
    """Save the best genome, its fitness metrics, and the trained DT model to files"""
    import pickle
    import os
    
    try:
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
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', 
                   exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save DT model if provided
        if dt_model is not None:
            model_filename = filename.replace('.json', '_dt_model.pkl')
            with open(model_filename, 'wb') as f:
                pickle.dump(dt_model, f)
            print(f"DT model saved to: {model_filename}")
        
        print(f"Genome saved to: {filename}")
        
    except IOError as e:
        print(f"Error saving genome: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error saving genome: {e}")
        raise
```

### 5. **Problema de Performance: Avaliação Sequencial**

**Problema:** A avaliação de fitness é feita sequencialmente, sem paralelização.

**Localização:** `feature_extractions/cgp.py:213, 226`

**Solução:** Usar multiprocessing ou joblib
```python
from joblib import Parallel, delayed

def compute_fitness_parallel(genome, images, labels, is_training=True):
    """Versão paralela do compute_fitness"""
    features_array = np.stack([evaluate(genome, image) for image in images])
    
    if is_training:
        model = type(eval_model)()
        model.fit(features_array, labels)
        return model.score(features_array, labels), model
    else:
        return eval_model.score(features_array, labels)

# Na função evolve:
fitness_results = Parallel(n_jobs=-1)(
    delayed(compute_fitness_parallel)(genome, train_images, train_labels) 
    for genome in population
)
```

### 6. **Falta de Validação de Parâmetros Globais**

**Problema:** Variáveis globais podem estar `None` quando usadas.

**Localização:** `feature_extractions/cgp.py:9-23`

**Solução:** Adicionar função de validação
```python
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

# Usar antes de operações críticas
def evaluate(genome, image):
    validate_config()
    # ... resto do código
```

### 7. **Problema de Design: Variáveis Globais**

**Problema:** Uso excessivo de variáveis globais dificulta testes e manutenção.

**Solução:** Criar classe de configuração
```python
class CGPConfig:
    def __init__(self, input_shape, functions, n_columns=20, n_rows=5, 
                 n_back=10, n_outputs=16):
        self.input_shape = input_shape
        self.functions = functions
        self.n_columns = n_columns
        self.n_rows = n_rows
        self.n_back = n_back
        self.n_outputs = n_outputs
        self.genes_per_node = 4
        self.n_nodes = n_columns * n_rows
    
    def validate(self):
        if self.functions is None or len(self.functions) == 0:
            raise ValueError("Functions must be provided")
        if self.n_nodes <= 0:
            raise ValueError("Invalid grid size")
        return True

# Usar como:
config = CGPConfig((64, 64, 3), vision_functions())
config.validate()
```

---

## 🟢 Melhorias de Qualidade

### 8. **Melhorar Documentação e Type Hints**

**Problema:** Falta de type hints completos e docstrings detalhadas.

**Solução:**
```python
from typing import List, Tuple, Type, Optional, Dict, Any
import numpy as np

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
    
    Example:
        >>> features = evaluate(genome, image)
        >>> print(features.shape)
        (16,)
    """
    # ... implementação
```

### 9. **Adicionar Logging em vez de Prints**

**Problema:** Uso de `print()` dificulta controle de logs.

**Solução:**
```python
import logging

logger = logging.getLogger(__name__)

def save_best_genome(...):
    logger.info(f"Saving best genome to {filename}")
    # ...
    logger.debug(f"Genome saved with fitness: {train_fitness:.4f}")
```

### 10. **Melhorar Crossover - Evitar Crossover Ruim**

**Problema:** Crossover de ponto único pode quebrar estruturas do genoma.

**Solução:** Crossover uniforme ou por bloco
```python
def crossover_uniform(parent1: List[float], parent2: List[float]) -> List[float]:
    """Uniform crossover - melhor preserva estruturas"""
    child = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

def crossover_block(parent1: List[float], parent2: List[float]) -> List[float]:
    """Crossover por blocos de nós"""
    block_size = GENES_PER_NODE
    n_blocks = len(parent1) // block_size
    
    child = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        if random.random() < 0.5:
            child.extend(parent1[start:end])
        else:
            child.extend(parent2[start:end])
    
    # Crossover dos outputs
    output_start = n_blocks * block_size
    if random.random() < 0.5:
        child.extend(parent1[output_start:])
    else:
        child.extend(parent2[output_start:])
    
    return child
```

### 11. **Adicionar Early Stopping**

**Problema:** Evolução continua mesmo sem melhoria.

**Solução:**
```python
def evolve(..., early_stopping_patience: int = 50):
    best_val_fitness = 0
    patience_counter = 0
    
    for generation in range(n_generations):
        # ... código existente ...
        
        if current_val_fitness > best_val_fitness:
            best_val_fitness = current_val_fitness
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at generation {generation + 1}")
            break
```

### 12. **Melhorar Mutação - Taxa Adaptativa**

**Problema:** Taxa de mutação fixa pode não ser ótima.

**Solução:**
```python
def mutate_adaptive(genome: List[float], base_rate: float = 0.1, 
                    generation: int = 0, max_generations: int = 100) -> List[float]:
    """Mutação com taxa adaptativa"""
    # Reduzir taxa de mutação ao longo das gerações
    adaptive_rate = base_rate * (1 - generation / max_generations)
    return mutate(genome, adaptive_rate)
```

### 13. **Adicionar Métricas de Diversidade**

**Problema:** Não há monitoramento da diversidade da população.

**Solução:**
```python
def calculate_population_diversity(population: List[List[float]]) -> float:
    """Calcula diversidade média da população"""
    if len(population) < 2:
        return 0.0
    
    total_diff = 0.0
    comparisons = 0
    
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            diff = sum(abs(a - b) for a, b in zip(population[i], population[j]))
            total_diff += diff
            comparisons += 1
    
    return total_diff / comparisons if comparisons > 0 else 0.0

# Usar em evolve():
diversity = calculate_population_diversity(population)
evolution_history["population_diversity"].append(diversity)
```

### 14. **Validação de Formato de Imagem**

**Problema:** Não valida formato de entrada.

**Solução:**
```python
def validate_image(image: np.ndarray) -> bool:
    """Valida formato de imagem"""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be numpy array, got {type(image)}")
    
    if len(image.shape) != 3:
        raise ValueError(f"Image must be 3D (H, W, C), got shape {image.shape}")
    
    if image.shape != INPUT_SHAPE:
        raise ValueError(f"Image shape {image.shape} doesn't match INPUT_SHAPE {INPUT_SHAPE}")
    
    return True
```

### 15. **Melhorar Tratamento de Exceções em Funções de Visão**

**Problema:** Funções de visão podem falhar silenciosamente.

**Solução:** Adicionar try-except em `vision.py`
```python
def conv3x3(x: np.ndarray, param: float) -> np.ndarray:
    try:
        # ... código existente ...
    except Exception as e:
        logger.warning(f"Error in conv3x3: {e}, returning input")
        return x
```

---

## 📊 Melhorias de Performance

### 16. **Otimizar Avaliação com Batch Processing**

**Problema:** Avaliação imagem por imagem é lenta.

**Solução:**
```python
def evaluate_batch(genome: List[float], images: List[np.ndarray]) -> np.ndarray:
    """Avalia múltiplas imagens de uma vez"""
    # Processar todas as imagens em batch
    features_list = [evaluate(genome, img) for img in images]
    return np.array(features_list)
```

### 17. **Usar NumPy Vectorization**

**Problema:** Loops Python são lentos.

**Solução:** Vectorizar operações onde possível
```python
# Em vez de:
features = []
for output_idx in output_genes:
    features.append(np.mean(node_outputs[int(output_idx)]))

# Usar:
output_indices = np.array([int(idx) for idx in output_genes])
features = np.array([np.mean(node_outputs[idx]) for idx in output_indices])
```

---

## 🔧 Melhorias de Manutenibilidade

### 18. **Separar Constantes em Arquivo de Configuração**

**Solução:** Criar `config.py`
```python
# config.py
class CGPConfig:
    DEFAULT_N_COLUMNS = 20
    DEFAULT_N_ROWS = 5
    DEFAULT_N_BACK = 10
    DEFAULT_N_OUTPUTS = 16
    DEFAULT_GENES_PER_NODE = 4
    DEFAULT_MUTATION_RATE = 0.1
    DEFAULT_POPULATION_SIZE = 50
    DEFAULT_N_GENERATIONS = 100
```

### 19. **Adicionar Testes Unitários**

**Solução:** Criar `test_cgp.py`
```python
import unittest
import numpy as np
from feature_extractions import cgp

class TestCGP(unittest.TestCase):
    def setUp(self):
        cgp.INPUT_SHAPE = (64, 64, 3)
        cgp.FUNCTIONS = vision_functions()
        cgp.N_NODES = cgp.get_n_nodes()
    
    def test_create_individual(self):
        genome = cgp.create_individual()
        self.assertEqual(len(genome), cgp.get_genome_length())
    
    def test_evaluate(self):
        genome = cgp.create_individual()
        image = np.random.rand(64, 64, 3)
        features = cgp.evaluate(genome, image)
        self.assertEqual(len(features), cgp.N_OUTPUTS)
```

### 20. **Adicionar Versionamento ao Modelo Salvo**

**Solução:**
```python
def save_best_genome(..., version: str = "1.0"):
    data = {
        "version": version,
        "genome": genome,
        # ... resto
    }
```

---

## 📝 Resumo de Prioridades

### Alta Prioridade (Implementar Primeiro)
1. ✅ Validação de entrada no `evaluate()` (#1)
2. ✅ Cache de fitness (#2)
3. ✅ Tratamento de erros I/O (#4)
4. ✅ Validação de configuração (#6)

### Média Prioridade
5. ✅ Paralelização (#5)
6. ✅ Early stopping (#11)
7. ✅ Melhorar documentação (#8)
8. ✅ Logging (#9)

### Baixa Prioridade (Melhorias Incrementais)
9. ✅ Refatorar variáveis globais (#7)
10. ✅ Métricas de diversidade (#13)
11. ✅ Testes unitários (#19)

---

## 🎯 Conclusão

O código está funcional, mas pode se beneficiar significativamente de:
- **Robustez**: Validação e tratamento de erros
- **Performance**: Cache, paralelização, otimizações
- **Manutenibilidade**: Refatoração, testes, documentação

Priorize as melhorias de alta prioridade para estabilidade e performance imediata.
