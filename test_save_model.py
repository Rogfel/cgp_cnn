#!/usr/bin/env python3
"""
Script de teste para verificar se o modelo DT está sendo salvo corretamente
"""

import numpy as np
from feature_extractions import cgp
from feature_extractions.vision import vision_functions
from classifications import DT

def test_save_and_load_model():
    """Testa o salvamento e carregamento do modelo DT"""
    
    print("=== Teste de Salvamento e Carregamento do Modelo DT ===\n")
    
    # Configuração inicial
    cgp.INPUT_SHAPE = (64, 64, 3)
    cgp.FUNCTIONS = vision_functions()
    cgp.N_NODES = cgp.get_n_nodes()
    
    print(f"N_OUTPUTS: {cgp.N_OUTPUTS}")
    print(f"N_NODES: {cgp.N_NODES}")
    print(f"Genome length: {cgp.get_genome_length()}")
    print()
    
    # Criar dados de teste pequenos
    n_samples = 20
    test_images = [np.random.rand(64, 64, 3) for _ in range(n_samples)]
    test_labels = [np.random.randint(0, 2) for _ in range(n_samples)]
    
    # Dividir em treino e validação
    train_size = int(0.7 * n_samples)
    train_images = test_images[:train_size]
    train_labels = test_labels[:train_size]
    val_images = test_images[train_size:]
    val_labels = test_labels[train_size:]
    
    print(f"Dados de teste: {n_samples} amostras")
    print(f"Treino: {len(train_images)} amostras")
    print(f"Validação: {len(val_images)} amostras")
    print()
    
    # Evoluir CGP com poucas gerações para teste
    print("Iniciando evolução CGP...")
    best_genome, best_fitness, val_fitness = cgp.evolve(
        train_images, train_labels,
        val_images, val_labels,
        n_generations=3,  # Poucas gerações para teste
        population_size=4,
        eval_model=DT.classification_model(),
        mutation_rate=0.2
    )
    
    print(f"\nEvolução concluída!")
    print(f"Best training fitness: {best_fitness:.4f}")
    print(f"Validation fitness: {val_fitness:.4f}")
    print()
    
    # Testar carregamento do modelo
    print("Testando carregamento do modelo...")
    try:
        genome_data, dt_model = cgp.load_best_genome_and_model()
        
        if dt_model is not None:
            print("✓ Modelo DT carregado com sucesso!")
            
            # Testar predição
            print("Testando predição...")
            test_image = np.random.rand(64, 64, 3)
            prediction, proba = cgp.predict_with_saved_model(genome_data, dt_model, test_image)
            
            print(f"✓ Predição realizada com sucesso!")
            print(f"Predição: {prediction}")
            print(f"Probabilidades: {proba}")
            
        else:
            print("✗ Modelo DT não foi encontrado!")
            
    except Exception as e:
        print(f"✗ Erro ao carregar modelo: {e}")
    
    print("\n=== Teste concluído ===")

if __name__ == "__main__":
    test_save_and_load_model()
