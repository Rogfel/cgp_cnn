#!/usr/bin/env python3
"""
Script de exemplo para usar o modelo CGP + DT salvo
"""

import numpy as np
from feature_extractions import cgp
from feature_extractions.vision import vision_functions
from dataset import load

def load_and_use_model():
    """Carrega e usa o modelo salvo para fazer predições"""
    
    print("=== Usando Modelo CGP + DT Salvo ===\n")
    
    # Configuração inicial (deve ser a mesma do treinamento)
    cgp.INPUT_SHAPE = (64, 64, 3)
    cgp.FUNCTIONS = vision_functions()
    cgp.N_NODES = cgp.get_n_nodes()
    
    print(f"Configuração:")
    print(f"  N_OUTPUTS: {cgp.N_OUTPUTS}")
    print(f"  N_NODES: {cgp.N_NODES}")
    print(f"  INPUT_SHAPE: {cgp.INPUT_SHAPE}")
    print()
    
    # Carregar modelo salvo
    try:
        genome_data, dt_model = cgp.load_best_genome_and_model()
        
        if dt_model is None:
            print("❌ Modelo DT não encontrado!")
            return
        
        print("✅ Modelo carregado com sucesso!")
        print(f"  Training fitness: {genome_data['training_fitness']:.4f}")
        print(f"  Validation fitness: {genome_data['validation_fitness']:.4f}")
        print()
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return
    
    # Carregar algumas imagens de teste
    print("Carregando imagens de teste...")
    try:
        load.PATH_DATASET = 'dataset/PetImages/'
        images, labels, classes = load.data(data_type='val')
        
        print(f"✅ Carregadas {len(images)} imagens de validação")
        print(f"  Classes: {classes}")
        print()
        
    except Exception as e:
        print(f"❌ Erro ao carregar imagens: {e}")
        print("Usando imagens aleatórias para teste...")
        images = [np.random.rand(64, 64, 3) for _ in range(5)]
        labels = [0, 1, 0, 1, 0]
        classes = ['Cat', 'Dog']
    
    # Fazer predições
    print("Fazendo predições...")
    correct_predictions = 0
    total_predictions = min(10, len(images))  # Testar apenas 10 imagens
    
    for i in range(total_predictions):
        try:
            # Fazer predição
            prediction, proba = cgp.predict_with_saved_model(genome_data, dt_model, images[i])
            
            # Verificar se a predição está correta
            is_correct = prediction == labels[i]
            if is_correct:
                correct_predictions += 1
            
            # Mostrar resultado
            confidence = max(proba) * 100
            status = "✅" if is_correct else "❌"
            
            print(f"Imagem {i+1}: {status}")
            print(f"  Predição: {classes[prediction]} (confiança: {confidence:.1f}%)")
            print(f"  Real: {classes[labels[i]]}")
            print(f"  Probabilidades: {[f'{p*100:.1f}%' for p in proba]}")
            print()
            
        except Exception as e:
            print(f"❌ Erro na predição da imagem {i+1}: {e}")
    
    # Mostrar estatísticas finais
    accuracy = (correct_predictions / total_predictions) * 100
    print("=== Estatísticas Finais ===")
    print(f"Predições corretas: {correct_predictions}/{total_predictions}")
    print(f"Acurácia: {accuracy:.1f}%")
    print()
    
    # Mostrar informações do modelo DT
    print("=== Informações do Modelo DT ===")
    print(f"Profundidade máxima: {dt_model.max_depth}")
    print(f"Número de features: {dt_model.n_features_in_}")
    print(f"Classes: {dt_model.classes_}")
    print(f"Número de nós: {dt_model.tree_.node_count}")
    print()

def analyze_feature_importance():
    """Analisa a importância das features extraídas pelo CGP"""
    
    print("=== Análise de Importância das Features ===\n")
    
    try:
        genome_data, dt_model = cgp.load_best_genome_and_model()
        
        if dt_model is None:
            print("❌ Modelo não encontrado!")
            return
        
        # Obter importância das features
        feature_importance = dt_model.feature_importances_
        
        print("Top 10 features mais importantes:")
        # Ordenar por importância
        sorted_indices = np.argsort(feature_importance)[::-1]
        
        for i in range(min(10, len(feature_importance))):
            idx = sorted_indices[i]
            importance = feature_importance[idx]
            print(f"  Feature {idx}: {importance:.4f}")
        
        print(f"\nTotal de features: {len(feature_importance)}")
        print(f"Features com importância > 0: {np.sum(feature_importance > 0)}")
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")

if __name__ == "__main__":
    load_and_use_model()
    print("\n" + "="*50 + "\n")
    analyze_feature_importance()
