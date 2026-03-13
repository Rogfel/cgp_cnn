"""
Pipeline de CGP com Aprendizado Ativo.

Este pipeline demonstra como usar aprendizado ativo para:
1. Comecar com um pequeno conjunto de dados rotulados
2. Iterativamente selecionar as amostras mais informativas
3. Adicionar pseudo-labels gerados pelo CGP
4. Evoluir o modelo progressivamente
"""

from dataset import load
from feature_extractions import cgp
from feature_extractions.vision import vision_functions
from classifications import DT
import numpy as np


def create_active_learning_pipeline():
    """
    Cria um pipeline de aprendizado ativo.
    
    Returns:
        Função que executa o AL com CGP
    """
    def run_pipeline(
        initial_split: float = 0.1,   # % inicial de dados rotulados
        unlabeled_split: float = 0.8,  # % de dados nao rotulados
        n_al_iterations: int = 50,
        samples_per_iteration: int = 30,
        query_strategy: str = "uncertainty"
    ):
        """
        Executa o pipeline de aprendizado ativo.
        """
        # 1. Carregar dataset completo
        print('***** Carregando dataset...')
        load.PATH_DATASET = 'dataset/PetImages/'
        all_images, all_labels, classes = load.data(data_type='train')
        
        n_total = len(all_images)
        print(f'Total de imagens: {n_total}')
        
        # 2. Dividir em: treino inicial, pool nao rotulado, validacao
        n_initial = int(n_total * initial_split)
        n_unlabeled = int(n_total * unlabeled_split)
        
        # Embaralhar
        indices = np.random.permutation(n_total)
        
        initial_indices = indices[:n_initial]
        unlabeled_indices = indices[n_initial:n_initial + n_unlabeled]
        val_indices = indices[n_initial + n_unlabeled:]
        
        initial_images = [all_images[i] for i in initial_indices]
        initial_labels = [all_labels[i] for i in initial_indices]
        
        unlabeled_images = [all_images[i] for i in unlabeled_indices]
        
        val_images = [all_images[i] for i in val_indices]
        val_labels = [all_labels[i] for i in val_indices]
        
        print('\n=== Divisao dos Dados ===')
        print(f'Treino inicial: {len(initial_images)} amostras')
        print(f'Pool nao rotulado: {len(unlabeled_images)} amostras')
        print(f'Validacao: {len(val_images)} amostras')
        
        # 3. Configurar CGP
        print('\n***** Configurando CGP...')
        cgp.INPUT_SHAPE = (load.IMG_HEIGHT, load.IMG_WIDTH, 3)
        cgp.FUNCTIONS = vision_functions()
        cgp.N_NODES = cgp.get_n_nodes()
        
        # 4. Executar Aprendizado Ativo
        print('\n***** Iniciando Aprendizado Ativo...')
        print(f'Estrategia de query: {query_strategy}')
        print(f'Iteracoes: {n_al_iterations}')
        print(f'Amostras por iteracao: {samples_per_iteration}')
        
        best_genome, best_fitness, val_fitness, history = cgp.evolve_active_learning(
            train_images=initial_images,
            train_labels=initial_labels,
            val_images=val_images,
            val_labels=val_labels,
            unlabeled_images=unlabeled_images,
            eval_model=DT.classification_model(),
            n_al_iterations=n_al_iterations,
            samples_per_iteration=samples_per_iteration,
            n_generations_per_al=30,
            population_size=20,
            mutation_rate=0.2,
            early_stopping_patience=30,
            query_strategy=query_strategy,
            save_models_dir="saved_models_al",
        )
        
        # 5. Resultados Finais
        print('\n' + '='*50)
        print('RESULTADOS FINAIS')
        print('='*50)
        print(f'Total de amostras rotuladas: {history["n_train_samples"][-1]}')
        print(f'Fitness treino final: {history["final_train_fitness"]:.4f}')
        print(f'Fitness validacao final: {history["final_val_fitness"]:.4f}')
        
        return best_genome, history
    
    return run_pipeline


if __name__ == "__main__":
    # Criar e executar pipeline
    run_al = create_active_learning_pipeline()
    
    # Executar com estrategia de incerteza
    print("="*60)
    print("EXECUTANDO COM ESTRATEGIA DE INCERTEZA")
    print("="*60)
    
    genome, history = run_al(
        initial_split=0.05,   # Apenas 5% rotulado inicialmente
        unlabeled_split=0.80,   # 80% nao rotulado
        n_al_iterations=10,
        samples_per_iteration=10,
        query_strategy="uncertainty"
    )
