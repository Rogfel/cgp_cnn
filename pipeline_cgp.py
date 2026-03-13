from dataset import load
from feature_extractions import cgp
from feature_extractions.vision import vision_functions
from classifications import DT
from classifications import random_forests as RF
from sklearn.model_selection import train_test_split
import cv2
import numpy as np


def convert_images_to_hsv(images, channel='HSV'):
    """
    Converte imagens para o espaço de cores especificado.
    
    Args:
        images: Lista de imagens RGB
        channel: 'HSV' para HSV completo, 'H' para só Hue, 'S' para só Saturação
    
    Returns:
        Lista de imagens convertidas
    """
    converted = []
    for img in images:
        if isinstance(img, np.ndarray):
            # Garantir que está no formato correto
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            
            if len(img.shape) == 2:
                # Imagem em escala de cinza, apenas retornar
                converted.append(img)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                if channel == 'HSV':
                    converted.append(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
                elif channel == 'H':
                    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                    converted.append(hsv[..., 0])
                elif channel == 'S':
                    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                    converted.append(hsv[..., 1])
                else:
                    converted.append(img)
            else:
                converted.append(img)
    return converted


# ============= CONFIGURAÇÃO =============
# Escolha o espaço de cores: 'RGB', 'HSV', 'H', ou 'S'
COLOR_SPACE = 'RGB'  # <- Mude para 'HSV', 'H' ou 'S' se quiser usar esses canais
# ======================================

# load dataset
print('*****1/4 Load images dataset')

load.PATH_DATASET = 'dataset/PetImages/'
images, labels, classes = load.data(data_type='train')

# Converter para o espaço de cores configurado
if COLOR_SPACE != 'RGB':
    print(f'*****1b/4 Convertendo imagens para {COLOR_SPACE}')
    images = convert_images_to_hsv(images, channel=COLOR_SPACE)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

# feature extraction
print('*****2/4 Init feature extraction')

# Configurar o CGP baseado no espaço de cores
n_channels = 1 if COLOR_SPACE in ['H', 'S'] else 3
cgp.INPUT_SHAPE = (load.IMG_HEIGHT, load.IMG_WIDTH, n_channels)
cgp.FUNCTIONS = vision_functions()
cgp.N_NODES = cgp.get_n_nodes()

# Evolve CGP (cache de fitness, paralelização e early stopping opcionais)
best_genome, best_fitness, val_fitness = cgp.evolve(
    train_images=X_train,
    train_labels=y_train,
    val_images=X_val,
    val_labels=y_val,
    n_generations=500,
    population_size=50,
    eval_model=DT.classification_model(),
    mutation_rate=0.15,
    n_jobs=1,                   # Parallel requires global config to be accessible in workers
    early_stopping_patience=30, # parar se val não melhorar em N gerações (None=desligado)
)

print(f"Training completed.")
print(f"Best training fitness: {best_fitness:.4f}")
print(f"Validation fitness: {val_fitness:.4f}")
print(f"Best genome: {best_genome}")