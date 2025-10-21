#!/usr/bin/env python3
"""
Script para gerar diagrama CGP usando NetworkX com Graphviz para layouts profissionais
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import graphviz
from matplotlib.patches import Circle, FancyBboxPatch
import sys
import os
from feature_extractions import cgp

# Adicionar o diretório atual ao path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_extractions.vision import vision_functions


class CGPGraphvizDiagram:
    def __init__(self, genome_file: str = "best_genome.json"):
        """Inicializa o gerador de diagrama com Graphviz"""
        self.genome_file = genome_file
        self.functions = vision_functions()
        self.function_names = [func.name for func in self.functions]
        
        # Parâmetros do CGP
        self.n_columns = cgp.N_COLUMNS
        self.n_rows = cgp.N_ROWS
        self.n_back = cgp.N_BACK
        self.n_outputs = cgp.N_OUTPUTS
        self.genes_per_node = cgp.GENES_PER_NODE
        
        self.n_nodes = self.n_columns * self.n_rows
        self.genome = None
        
        # Grafo NetworkX
        self.G = nx.DiGraph()
        
        self.load_genome()
    
    def load_genome(self):
        """Carrega o genoma do arquivo JSON"""
        try:
            with open(self.genome_file, 'r') as f:
                data = json.load(f)
                self.genome = data['genome']
                
                if 'hyperparameters' in data:
                    self.n_columns = data['hyperparameters'].get('n_columns', self.n_columns)
                    self.n_rows = data['hyperparameters'].get('n_rows', self.n_rows)
                    self.n_back = data['hyperparameters'].get('n_back', self.n_back)
                    self.n_outputs = data['hyperparameters'].get('n_outputs', self.n_outputs)
                    self.genes_per_node = data['hyperparameters'].get('genes_per_node', self.genes_per_node)
                    
                    self.n_nodes = self.n_columns * self.n_rows
                
                print(f"Genoma carregado: {len(self.genome)} genes")
                
        except FileNotFoundError:
            print(f"Arquivo {self.genome_file} não encontrado!")
            sys.exit(1)
        except Exception as e:
            print(f"Erro ao carregar genoma: {e}")
            sys.exit(1)
    
    def parse_genome(self):
        """Decodifica o genoma em estrutura de nós"""
        nodes = {}
        outputs = []
        
        # Parse dos nós
        for i in range(self.n_nodes):
            idx = i * self.genes_per_node
            func_id = int(self.genome[idx])
            input1_idx = int(self.genome[idx + 1])
            input2_idx = int(self.genome[idx + 2])
            param = self.genome[idx + 3]
            
            node_idx = i + 3  # +3 para RGB channels
            
            nodes[node_idx] = {
                'function_id': func_id,
                'function_name': self.function_names[func_id],
                'input1': input1_idx,
                'input2': input2_idx,
                'parameter': param,
                'n_inputs': self.functions[func_id].n_inputs
            }
        
        # Parse das saídas
        output_start = self.n_nodes * self.genes_per_node
        for i in range(self.n_outputs):
            output_idx = int(self.genome[output_start + i])
            outputs.append(output_idx)
        
        return {'nodes': nodes, 'outputs': outputs}
    
    def find_active_nodes(self, parsed_genome):
        """Encontra nós ativos (conectados às saídas)"""
        active_nodes = set()
        to_process = set(parsed_genome['outputs'])
        
        while to_process:
            current = to_process.pop()
            if current in active_nodes or current < 3:
                continue
                
            active_nodes.add(current)
            
            if current in parsed_genome['nodes']:
                node = parsed_genome['nodes'][current]
                if node['input1'] >= 3:
                    to_process.add(node['input1'])
                if node['n_inputs'] == 2 and node['input2'] >= 3:
                    to_process.add(node['input2'])
        
        return active_nodes
    
    def build_network_graph(self, parsed_genome, active_nodes):
        """Constrói o grafo NetworkX"""
        self.G.clear()
        
        # Adiciona nós de entrada
        used_inputs = set()
        for active_node in active_nodes:
            if active_node in parsed_genome['nodes']:
                node_data = parsed_genome['nodes'][active_node]
                if node_data['input1'] < 3:
                    used_inputs.add(node_data['input1'])
                if node_data['n_inputs'] == 2 and node_data['input2'] < 3:
                    used_inputs.add(node_data['input2'])
        
        # Adiciona nós de entrada ao grafo
        for input_id in used_inputs:
            self.G.add_node(input_id, 
                          node_type='input',
                          label=f'x{input_id}',
                          color='lightblue',
                          shape='circle',
                          style='filled')
        
        # Adiciona nós de processamento ativos
        for node_id in active_nodes:
            if node_id in parsed_genome['nodes']:
                node_data = parsed_genome['nodes'][node_id]
                self.G.add_node(node_id,
                              node_type='processing',
                              label=node_data['function_name'],
                              function_name=node_data['function_name'],
                              input1=node_data['input1'],
                              input2=node_data['input2'],
                              n_inputs=node_data['n_inputs'],
                              color='lightgreen',
                              shape='box',
                              style='filled,rounded')
        
        # Adiciona nós de saída
        for i, output_node in enumerate(parsed_genome['outputs']):
            if output_node in active_nodes or output_node < 3:
                self.G.add_node(f'output_{i}',
                              node_type='output',
                              label=f'out{i}',
                              connected_to=output_node,
                              color='lightcoral',
                              shape='doublecircle',
                              style='filled')
        
        # Adiciona arestas (conexões)
        for node_id in active_nodes:
            if node_id in parsed_genome['nodes']:
                node_data = parsed_genome['nodes'][node_id]
                
                # Conexão com input1
                self.G.add_edge(node_data['input1'], node_id, 
                              connection_type='input1',
                              color='blue',
                              penwidth=2)
                
                # Conexão com input2 (se aplicável)
                if node_data['n_inputs'] == 2:
                    self.G.add_edge(node_data['input2'], node_id,
                                  connection_type='input2',
                                  color='green',
                                  penwidth=2)
        
        # Conexões para saídas
        for i, output_node in enumerate(parsed_genome['outputs']):
            if output_node in active_nodes or output_node < 3:
                self.G.add_edge(output_node, f'output_{i}',
                              connection_type='output',
                              color='red',
                              penwidth=3,
                              style='dashed')
    
    def create_graphviz_diagram(self, save_path: str = "cgp_graphviz_diagram"):
        """Cria diagrama usando Graphviz"""
        parsed_genome = self.parse_genome()
        active_nodes = self.find_active_nodes(parsed_genome)
        
        # Constrói o grafo
        self.build_network_graph(parsed_genome, active_nodes)
        
        # Cria o grafo Graphviz
        dot = graphviz.Digraph(comment='CGP Circuit Diagram')
        dot.attr(rankdir='LR', size='20,14', dpi='300')
        dot.attr('node', fontsize='10', fontname='Arial')
        dot.attr('edge', fontsize='8', fontname='Arial')
        
        # Adiciona nós ao grafo Graphviz
        for node, data in self.G.nodes(data=True):
            if data['node_type'] == 'input':
                dot.node(str(node), 
                        f"{data['label']}\\n({node})",
                        fillcolor=data['color'],
                        shape=data['shape'],
                        style=data['style'])
            elif data['node_type'] == 'processing':
                dot.node(str(node), 
                        f"{data['function_name']}\\n({node})",
                        fillcolor=data['color'],
                        shape=data['shape'],
                        style=data['style'])
            elif data['node_type'] == 'output':
                dot.node(str(node), 
                        f"{data['label']}\\n({data['connected_to']})",
                        fillcolor=data['color'],
                        shape=data['shape'],
                        style=data['style'])
        
        # Adiciona arestas ao grafo Graphviz
        for source, target, data in self.G.edges(data=True):
            dot.edge(str(source), str(target),
                    color=data['color'],
                    penwidth=str(data['penwidth']),
                    style=data.get('style', 'solid'))
        
        # Salva em diferentes formatos
        dot.render(save_path, format='png', cleanup=True)
        dot.render(save_path, format='svg', cleanup=True)
        dot.render(save_path, format='pdf', cleanup=True)
        
        print(f"Diagrama Graphviz salvo em: {save_path}.png, {save_path}.svg, {save_path}.pdf")
        
        return dot
    
    def create_networkx_graphviz_diagram(self, save_path: str = "cgp_networkx_graphviz.png"):
        """Cria diagrama usando NetworkX com layout Graphviz"""
        parsed_genome = self.parse_genome()
        active_nodes = self.find_active_nodes(parsed_genome)
        
        # Constrói o grafo
        self.build_network_graph(parsed_genome, active_nodes)
        
        # Tenta usar layout Graphviz
        try:
            pos = nx.nx_agraph.graphviz_layout(self.G, prog='dot')
            print("Usando layout Graphviz (dot)")
        except:
            try:
                pos = nx.nx_agraph.graphviz_layout(self.G, prog='neato')
                print("Usando layout Graphviz (neato)")
            except:
                try:
                    pos = nx.nx_agraph.graphviz_layout(self.G, prog='fdp')
                    print("Usando layout Graphviz (fdp)")
                except:
                    print("Graphviz não disponível, usando layout spring")
                    pos = nx.spring_layout(self.G, k=3, iterations=50)
        
        # Cria figura
        fig, ax = plt.subplots(figsize=(20, 14))
        
        # Desenha arestas
        self._draw_networkx_edges(ax, pos)
        
        # Desenha nós
        self._draw_networkx_nodes(ax, pos)
        
        # Configurações do plot
        ax.set_xlim(min(x for x, y in pos.values()) - 1, max(x for x, y in pos.values()) + 1)
        ax.set_ylim(min(y for x, y in pos.values()) - 1, max(y for x, y in pos.values()) + 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.title("CGP Network Diagram - Graphviz Layout", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"Diagrama NetworkX+Graphviz salvo em: {save_path}")
    
    def _draw_networkx_edges(self, ax, pos):
        """Desenha arestas do NetworkX"""
        for edge in self.G.edges(data=True):
            source, target, data = edge
            if source in pos and target in pos:
                x1, y1 = pos[source]
                x2, y2 = pos[target]
                
                color = data.get('color', 'black')
                width = data.get('penwidth', 1)
                style = data.get('style', 'solid')
                
                if style == 'dashed':
                    linestyle = '--'
                else:
                    linestyle = '-'
                
                ax.plot([x1, x2], [y1, y2], linestyle, 
                       color=color, alpha=0.8, linewidth=width)
    
    def _draw_networkx_nodes(self, ax, pos):
        """Desenha nós do NetworkX"""
        for node, data in self.G.nodes(data=True):
            if node not in pos:
                continue
                
            x, y = pos[node]
            node_type = data['node_type']
            
            if node_type == 'input':
                self._draw_input_node_networkx(ax, x, y, node, data)
            elif node_type == 'processing':
                self._draw_processing_node_networkx(ax, x, y, node, data)
            elif node_type == 'output':
                self._draw_output_node_networkx(ax, x, y, node, data)
    
    def _draw_input_node_networkx(self, ax, x, y, node, data):
        """Desenha nó de entrada no NetworkX"""
        # Círculo do nó
        circle = Circle((x, y), 0.3, color='black', fill=True, 
                       facecolor=data['color'], linewidth=2)
        ax.add_patch(circle)
        
        # Label do nó
        ax.text(x, y, data['label'], ha='center', va='center', 
               fontsize=10, fontweight='bold')
    
    def _draw_processing_node_networkx(self, ax, x, y, node, data):
        """Desenha nó de processamento no NetworkX"""
        # Retângulo com bordas arredondadas
        rect = FancyBboxPatch((x - 0.4, y - 0.2), 0.8, 0.4,
                             boxstyle="round,pad=0.05",
                             facecolor=data['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Nome da função
        ax.text(x, y, data['function_name'], ha='center', va='center', 
               fontsize=9, fontweight='bold')
    
    def _draw_output_node_networkx(self, ax, x, y, node, data):
        """Desenha nó de saída no NetworkX"""
        # Círculo duplo
        outer_circle = Circle((x, y), 0.3, color='black', fill=False, linewidth=3)
        inner_circle = Circle((x, y), 0.2, color='black', fill=True, 
                             facecolor=data['color'], linewidth=2)
        ax.add_patch(outer_circle)
        ax.add_patch(inner_circle)
        
        # Label do nó
        ax.text(x, y, data['label'], ha='center', va='center', 
               fontsize=10, fontweight='bold')
    
    def get_network_info(self):
        """Retorna informações sobre a rede"""
        return {
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(),
            'input_nodes': len([n for n, d in self.G.nodes(data=True) if d['node_type'] == 'input']),
            'processing_nodes': len([n for n, d in self.G.nodes(data=True) if d['node_type'] == 'processing']),
            'output_nodes': len([n for n, d in self.G.nodes(data=True) if d['node_type'] == 'output'])
        }
    
    def export_graph_data(self, filename: str = "cgp_graphviz_data.json"):
        """Exporta dados do grafo para análise"""
        graph_data = {
            'nodes': dict(self.G.nodes(data=True)),
            'edges': list(self.G.edges(data=True)),
            'info': self.get_network_info()
        }
        
        with open(filename, 'w') as f:
            json.dump(graph_data, f, indent=2, default=str)
        
        print(f"Dados do grafo exportados para: {filename}")


def main():
    """Função principal"""
    if len(sys.argv) > 1:
        genome_file = sys.argv[1]
    else:
        genome_file = "best_genome.json"
    
    diagram = CGPGraphvizDiagram(genome_file)
    
    # Cria diagrama Graphviz puro
    print("Criando diagrama Graphviz puro...")
    dot = diagram.create_graphviz_diagram()
    
    # Cria diagrama NetworkX com layout Graphviz
    print("\nCriando diagrama NetworkX com layout Graphviz...")
    diagram.create_networkx_graphviz_diagram()
    
    # Mostra informações da rede
    info = diagram.get_network_info()
    print(f"\nInformações da rede:")
    print(f"- Total de nós: {info['nodes']}")
    print(f"- Total de arestas: {info['edges']}")
    print(f"- Nós de entrada: {info['input_nodes']}")
    print(f"- Nós de processamento: {info['processing_nodes']}")
    print(f"- Nós de saída: {info['output_nodes']}")
    
    # Exporta dados do grafo
    diagram.export_graph_data()


if __name__ == "__main__":
    main()
