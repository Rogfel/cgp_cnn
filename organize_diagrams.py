#!/usr/bin/env python3
"""
Script para organizar todos os diagramas existentes na pasta 'diagramas'
"""

import os
import shutil
import glob

def organize_existing_diagrams():
    """Move todos os diagramas existentes para a pasta diagramas"""
    
    # Criar pasta diagramas se não existir
    os.makedirs("diagramas", exist_ok=True)
    
    # Padrões de arquivos de diagramas
    patterns = [
        "cgp_*.png",
        "cgp_*.svg", 
        "cgp_*.pdf",
        "cgp_*.jpg",
        "cgp_*.jpeg"
    ]
    
    moved_files = []
    
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            if os.path.isfile(file):
                destination = os.path.join("diagramas", file)
                # Se já existe, adiciona sufixo
                if os.path.exists(destination):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(destination):
                        destination = os.path.join("diagramas", f"{base}_old_{counter}{ext}")
                        counter += 1
                
                shutil.move(file, destination)
                moved_files.append(f"{file} -> {destination}")
    
    return moved_files

def create_diagram_index():
    """Cria um índice dos diagramas na pasta"""
    
    index_path = os.path.join("diagramas", "README.md")
    
    with open(index_path, 'w') as f:
        f.write("# Diagramas CGP - Índice\n\n")
        f.write("Esta pasta contém todos os diagramas gerados para a rede CGP.\n\n")
        
        f.write("## Tipos de Diagramas\n\n")
        
        f.write("### 1. Matplotlib\n")
        f.write("- `cgp_matplotlib_diagram.png` - Diagrama original usando matplotlib\n\n")
        
        f.write("### 2. Graphviz - Layouts Profissionais\n")
        f.write("- `cgp_hierarchical.*` - Layout hierárquico (dot)\n")
        f.write("- `cgp_force_directed.*` - Layout força dirigida (neato)\n")
        f.write("- `cgp_circular.*` - Layout circular (circo)\n\n")
        
        f.write("### 3. NetworkX\n")
        f.write("- `cgp_networkx_spring.png` - Layout spring do NetworkX\n\n")
        
        f.write("### 4. Análise da Rede\n")
        f.write("- `network_analysis.json` - Dados da análise em JSON\n")
        f.write("- `network_report.txt` - Relatório textual da análise\n\n")
        
        f.write("## Formatos Disponíveis\n")
        f.write("- **PNG**: Imagens raster de alta qualidade\n")
        f.write("- **SVG**: Imagens vetoriais escaláveis\n")
        f.write("- **PDF**: Documentos vetoriais para impressão\n\n")
        
        f.write("## Estatísticas da Rede\n")
        f.write("- Total de nós: 143\n")
        f.write("- Total de arestas: 151\n")
        f.write("- Nós de entrada: 2\n")
        f.write("- Nós de processamento: 77\n")
        f.write("- Nós de saída: 64\n")
        f.write("- Funções utilizadas: 15 diferentes\n\n")
        
        f.write("## Como Gerar Novos Diagramas\n")
        f.write("```bash\n")
        f.write("python generate_all_diagrams.py\n")
        f.write("```\n\n")
        
        f.write("## Descrição dos Layouts\n\n")
        f.write("### Hierárquico (dot)\n")
        f.write("Organiza os nós em camadas verticais, mostrando claramente o fluxo de dados da entrada para a saída.\n\n")
        
        f.write("### Força Dirigida (neato)\n")
        f.write("Usa um algoritmo de força física para posicionar os nós de forma natural, evitando sobreposições.\n\n")
        
        f.write("### Circular (circo)\n")
        f.write("Organiza os nós em um círculo, útil para visualizar redes complexas com muitas conexões.\n\n")
        
        f.write("### Spring (NetworkX)\n")
        f.write("Algoritmo de mola que posiciona os nós baseado em forças de atração e repulsão.\n\n")

def main():
    """Função principal"""
    print("=== Organizando diagramas existentes ===\n")
    
    # Move arquivos existentes
    moved_files = organize_existing_diagrams()
    
    if moved_files:
        print("Arquivos movidos:")
        for move in moved_files:
            print(f"  ✓ {move}")
    else:
        print("Nenhum arquivo de diagrama encontrado para mover.")
    
    print()
    
    # Cria índice
    create_diagram_index()
    print("✓ Índice criado em: diagramas/README.md")
    
    print("\n=== Organização concluída! ===")
    print("Todos os diagramas estão agora organizados na pasta 'diagramas/'")

if __name__ == "__main__":
    main()
