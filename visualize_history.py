"""
Visualização do Histórico de Evolução CGP.

Este script gera gráficos para visualizar:
1. Melhor fitness por geração (treino e validação)
2. Evolução da população ao longo das gerações
3. Diferença entre treino e validação (overfitting)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_history(json_path: str = "evolution_history_old.json") -> dict:
    """Carrega o histórico de evolução do arquivo JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_best_fitness(history: dict, save_path: str = None):
    """
    Plota o melhor fitness por geração para treino e validação.
    """
    generations = history['generations']
    train_fitness = history['best_training_fitness']
    val_fitness = history['best_validation_fitness']
    
    # Remover duplicatas para gráficos mais limpos
    # (o JSON pode ter valores repetidos para mesmas gerações)
    unique_generations = []
    unique_train = []
    unique_val = []
    
    seen = set()
    for g, t, v in zip(generations, train_fitness, val_fitness):
        if g not in seen:
            seen.add(g)
            unique_generations.append(g)
            unique_train.append(t)
            unique_val.append(v)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(unique_generations, unique_train, 'b-', linewidth=2, 
             label='Treino', marker='o', markersize=4)
    plt.plot(unique_generations, unique_val, 'r-', linewidth=2, 
             label='Validação', marker='s', markersize=4)
    
    plt.xlabel('Geração', fontsize=12)
    plt.ylabel('Fitness (Acurácia)', fontsize=12)
    plt.title('Evolução do Melhor Fitness por Geração', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Adicionar área entre treino e validação
    plt.fill_between(unique_generations, unique_train, unique_val, 
                     alpha=0.2, color='gray', label='Gap')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")
    
    plt.show()


def plot_population_evolution(history: dict, save_path: str = None):
    """
    Plota a evolução de toda a população ao longo das gerações.
    Mostra mínimo, médio e máximo fitness da população.
    """
    generations = history['generations']
    population_fitness = history['population_training_fitness']
    
    # Calcular estatísticas da população por geração
    pop_stats = {
        'min': [],
        'mean': [],
        'max': []
    }
    
    # Agrupar por geração (pode haver múltiplas entradas por geração)
    gen_data = {}
    for g, pop in zip(generations, population_fitness):
        if g not in gen_data:
            gen_data[g] = []
        gen_data[g].extend(pop)
    
    for g in sorted(gen_data.keys()):
        data = gen_data[g]
        pop_stats['min'].append(np.min(data))
        pop_stats['mean'].append(np.mean(data))
        pop_stats['max'].append(np.max(data))
    
    generations_unique = sorted(gen_data.keys())
    
    plt.figure(figsize=(12, 6))
    
    # Plotar área de população
    plt.fill_between(generations_unique, pop_stats['min'], pop_stats['max'], 
                     alpha=0.3, color='blue', label='Min-Max População')
    
    # Plotar média da população
    plt.plot(generations_unique, pop_stats['mean'], 'b-', linewidth=2, 
             label='Média População', marker='o', markersize=4)
    
    # Plotar melhor indivíduo
    plt.plot(generations_unique, pop_stats['max'], 'g-', linewidth=2.5, 
             label='Melhor Indivíduo', marker='^', markersize=5)
    
    plt.xlabel('Geração', fontsize=12)
    plt.ylabel('Fitness (Acurácia)', fontsize=12)
    plt.title('Evolução da População CGP', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")
    
    plt.show()


def plot_overfitting(history: dict, save_path: str = None):
    """
    Plota a diferença entre fitness de treino e validação para detectar overfitting.
    """
    generations = history['generations']
    train_fitness = history['best_training_fitness']
    val_fitness = history['best_validation_fitness']
    
    # Calcular gap (overfitting)
    gap = [t - v for t, v in zip(train_fitness, val_fitness)]
    
    # Remover duplicatas
    unique_data = {}
    for g, t, v, g_val in zip(generations, train_fitness, val_fitness, gap):
        if g not in unique_data:
            unique_data[g] = {'train': t, 'val': v, 'gap': g_val}
    
    generations_unique = sorted(unique_data.keys())
    train_unique = [unique_data[g]['train'] for g in generations_unique]
    val_unique = [unique_data[g]['val'] for g in generations_unique]
    gap_unique = [unique_data[g]['gap'] for g in generations_unique]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Gráfico superior: fitness
    ax1.plot(generations_unique, train_unique, 'b-', linewidth=2, 
             label='Treino', marker='o', markersize=4)
    ax1.plot(generations_unique, val_unique, 'r-', linewidth=2, 
             label='Validação', marker='s', markersize=4)
    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.set_title('Fitness Treino vs Validação', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Gráfico inferior: gap
    ax2.bar(generations_unique, gap_unique, color='orange', alpha=0.7, edgecolor='darkorange')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Geração', fontsize=12)
    ax2.set_ylabel('Gap (Train - Val)', fontsize=12)
    ax2.set_title('Diferença Treino-Validação (Overfitting)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")
    
    plt.show()


def plot_convergence_speed(history: dict, save_path: str = None):
    """
    Plota a velocidade de convergência mostrando quantas vezes o melhor
    fitness melhorou ao longo das gerações.
    """
    generations = history['generations']
    train_fitness = history['best_training_fitness']
    
    # Obter melhores valores únicos
    unique_data = {}
    for g, t in zip(generations, train_fitness):
        if g not in unique_data or t > unique_data[g]:
            unique_data[g] = t
    
    generations_unique = sorted(unique_data.keys())
    fitness_unique = [unique_data[g] for g in generations_unique]
    
    # Calcular melhorias
    improvements = []
    current_best = 0
    for f in fitness_unique:
        if f > current_best:
            improvements.append(f - current_best)
            current_best = f
        else:
            improvements.append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico esquerdo: fitness acumulado
    ax1.plot(generations_unique, fitness_unique, 'g-', linewidth=2, 
             marker='o', markersize=4)
    ax1.set_xlabel('Geração', fontsize=12)
    ax1.set_ylabel('Melhor Fitness', fontsize=12)
    ax1.set_title('Convergência do Melhor Fitness', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Adicionar anotações de melhoria
    for i, (g, f, imp) in enumerate(zip(generations_unique, fitness_unique, improvements)):
        if imp > 0:
            ax1.annotate(f'+{imp:.3f}', (g, f), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=8, color='green')
    
    # Gráfico direito: melhorias por geração
    colors = ['green' if imp > 0 else 'lightgray' for imp in improvements]
    ax2.bar(generations_unique, improvements, color=colors, edgecolor='darkgreen', alpha=0.7)
    ax2.set_xlabel('Geração', fontsize=12)
    ax2.set_ylabel('Melhoria', fontsize=12)
    ax2.set_title('Melhorias por Geração', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")
    
    plt.show()


def plot_summary_dashboard(history: dict, save_path: str = None):
    """
    Cria um dashboard completo com múltiplas visualizações.
    """
    fig = plt.figure(figsize=(16, 12))
    
    generations = history['generations']
    train_fitness = history['best_training_fitness']
    val_fitness = history['best_validation_fitness']
    population_fitness = history['population_training_fitness']
    
    # Preparar dados únicos
    unique_data = {}
    for g, t, v in zip(generations, train_fitness, val_fitness):
        if g not in unique_data:
            unique_data[g] = {'train': t, 'val': v}
    
    generations_unique = sorted(unique_data.keys())
    train_unique = [unique_data[g]['train'] for g in generations_unique]
    val_unique = [unique_data[g]['val'] for g in generations_unique]
    
    # 1. Melhor fitness (topo)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(generations_unique, train_unique, 'b-', linewidth=2, label='Treino')
    ax1.plot(generations_unique, val_unique, 'r-', linewidth=2, label='Validação')
    ax1.fill_between(generations_unique, train_unique, val_unique, alpha=0.2, color='gray')
    ax1.set_xlabel('Geração')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Melhor Fitness por Geração', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. População (topo direito)
    ax2 = fig.add_subplot(2, 2, 2)
    
    gen_data = {}
    for g, pop in zip(generations, population_fitness):
        if g not in gen_data:
            gen_data[g] = []
        gen_data[g].extend(pop)
    
    pop_means = [np.mean(gen_data[g]) for g in sorted(gen_data.keys())]
    pop_maxs = [np.max(gen_data[g]) for g in sorted(gen_data.keys())]
    pop_mins = [np.min(gen_data[g]) for g in sorted(gen_data.keys())]
    gens = sorted(gen_data.keys())
    
    ax2.fill_between(gens, pop_mins, pop_maxs, alpha=0.3, color='blue', label='Min-Max')
    ax2.plot(gens, pop_means, 'b-', linewidth=2, label='Média')
    ax2.plot(gens, pop_maxs, 'g-', linewidth=2, label='Melhor')
    ax2.set_xlabel('Geração')
    ax2.set_ylabel('Fitness')
    ax2.set_title('Evolução da População', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Gap de overfitting (baixo esquerdo)
    ax3 = fig.add_subplot(2, 2, 3)
    gaps = [t - v for t, v in zip(train_unique, val_unique)]
    ax3.bar(generations_unique, gaps, color='orange', alpha=0.7, edgecolor='darkorange')
    ax3.axhline(y=np.mean(gaps), color='red', linestyle='--', linewidth=2, 
                label=f'Média Gap: {np.mean(gaps):.3f}')
    ax3.set_xlabel('Geração')
    ax3.set_ylabel('Gap (Train - Val)')
    ax3.set_title('Diferença Treino-Validação', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Estatísticas finais (baixo direito)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Calcular estatísticas
    final_train = train_unique[-1]
    final_val = val_unique[-1]
    best_train_idx = np.argmax(train_unique)
    best_train = train_unique[best_train_idx]
    best_val_idx = np.argmax(val_unique)
    best_val = val_unique[best_val_idx]
    
    stats_text = f"""
    ╔══════════════════════════════════════════╗
    ║          ESTATÍSTICAS FINAIS              ║
    ╠══════════════════════════════════════════╣
    ║ Gerações Totais: {len(generations_unique):>25} ║
    ║                                          ║
    ║ Treino Final:    {final_train:>25.4f} ║
    ║ Validação Final: {final_val:>25.4f} ║
    ║                                          ║
    ║ Melhor Treino:   {best_train:>25.4f} ║
    ║   (Geração {best_train_idx + 1})                      ║
    ║                                          ║
    ║ Melhor Validação: {best_val:>25.4f} ║
    ║   (Geração {best_val_idx + 1})                      ║
    ║                                          ║
    ║ Gap Médio:       {np.mean(gaps):>25.4f} ║
    ╚══════════════════════════════════════════╝
    """
    
    ax4.text(0.5, 0.5, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Dashboard - Evolução CGP', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard salvo em: {save_path}")
    
    plt.show()


def main():
    """Função principal para gerar todas as visualizações."""
    # Carregar histórico
    print("Carregando histórico de evolução...")
    history = load_history()
    
    print(f"\nEstatísticas do histórico:")
    print(f"  - Gerações: {len(history['generations'])}")
    print(f"  - Populações registradas: {len(history['population_training_fitness'])}")
    
    # Criar diretório para salvar gráficos
    output_dir = Path("visualization_output")
    output_dir.mkdir(exist_ok=True)
    
    # Gerar todos os gráficos
    print("\n1. Gerando gráfico de melhor fitness...")
    plot_best_fitness(history, save_path=str(output_dir / "best_fitness.png"))
    
    print("2. Gerando gráfico de evolução da população...")
    plot_population_evolution(history, save_path=str(output_dir / "population_evolution.png"))
    
    print("3. Gerando gráfico de overfitting...")
    plot_overfitting(history, save_path=str(output_dir / "overfitting.png"))
    
    print("4. Gerando gráfico de convergência...")
    plot_convergence_speed(history, save_path=str(output_dir / "convergence.png"))
    
    print("5. Gerando dashboard completo...")
    plot_summary_dashboard(history, save_path=str(output_dir / "dashboard.png"))
    
    print(f"\n✓ Todos os gráficos salvos em: {output_dir}/")
    print("\nGráficos gerados:")
    print("  - best_fitness.png        : Melhor fitness treino vs validação")
    print("  - population_evolution.png: Evolução da população (min/média/max)")
    print("  - overfitting.png         : Diferença treino-validação")
    print("  - convergence.png         : Velocidade de convergência")
    print("  - dashboard.png           : Dashboard completo com tudo")


if __name__ == "__main__":
    main()
