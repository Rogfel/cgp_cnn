# Diagramas CGP - Índice

Esta pasta contém todos os diagramas gerados para a rede CGP.

## Tipos de Diagramas

### 1. Matplotlib
- `cgp_matplotlib_diagram.png` - Diagrama original usando matplotlib

### 2. Graphviz - Layouts Profissionais
- `cgp_hierarchical.*` - Layout hierárquico (dot)
- `cgp_force_directed.*` - Layout força dirigida (neato)
- `cgp_circular.*` - Layout circular (circo)

### 3. NetworkX
- `cgp_networkx_spring.png` - Layout spring do NetworkX

### 4. Análise da Rede
- `network_analysis.json` - Dados da análise em JSON
- `network_report.txt` - Relatório textual da análise

## Formatos Disponíveis
- **PNG**: Imagens raster de alta qualidade
- **SVG**: Imagens vetoriais escaláveis
- **PDF**: Documentos vetoriais para impressão

## Estatísticas da Rede
- Total de nós: 143
- Total de arestas: 151
- Nós de entrada: 2
- Nós de processamento: 77
- Nós de saída: 64
- Funções utilizadas: 15 diferentes

## Como Gerar Novos Diagramas
```bash
python generate_all_diagrams.py
```

## Descrição dos Layouts

### Hierárquico (dot)
Organiza os nós em camadas verticais, mostrando claramente o fluxo de dados da entrada para a saída.

### Força Dirigida (neato)
Usa um algoritmo de força física para posicionar os nós de forma natural, evitando sobreposições.

### Circular (circo)
Organiza os nós em um círculo, útil para visualizar redes complexas com muitas conexões.

### Spring (NetworkX)
Algoritmo de mola que posiciona os nós baseado em forças de atração e repulsão.

