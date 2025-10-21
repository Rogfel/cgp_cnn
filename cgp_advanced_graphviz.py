#!/usr/bin/env python3
"""
Script avançado para gerar diagramas CGP usando diferentes algoritmos de layout do Graphviz
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


class CGPAdvancedGraphviz:
    def __init__(self, genome_file: str = "best_genome.json"):
        """Inicializa o gerador de diagrama avançado com Graphviz"""
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
    
    def create_hierarchical_diagram(self, save_path: str = "cgp_hierarchical"):
        """Cria diagrama hierárquico usando Graphviz"""
        parsed_genome = self.parse_genome()
        active_nodes = self.find_active_nodes(parsed_genome)
        
        # Constrói o grafo
        self.build_network_graph(parsed_genome, active_nodes)
        
        # Cria o grafo Graphviz hierárquico
        dot = graphviz.Digraph(comment='CGP Hierarchical Diagram')
        dot.attr(rankdir='TB', size='20,14', dpi='300')
        dot.attr('node', fontsize='10', fontname='Arial')
        dot.attr('edge', fontsize='8', fontname='Arial')
        
        # Organiza nós em camadas
        layers = self._organize_nodes_in_layers()
        
        # Adiciona nós organizados por camadas
        for layer_idx, layer_nodes in enumerate(layers):
            with dot.subgraph(name=f'layer_{layer_idx}') as layer:
                layer.attr(rank='same')
                for node in layer_nodes:
                    if node in self.G.nodes:
                        data = self.G.nodes[node]
                        if data['node_type'] == 'input':
                            layer.node(str(node), 
                                    f"{data['label']}\\n({node})",
                                    fillcolor=data['color'],
                                    shape=data['shape'],
                                    style=data['style'])
                        elif data['node_type'] == 'processing':
                            layer.node(str(node), 
                                    f"{data['function_name']}\\n({node})",
                                    fillcolor=data['color'],
                                    shape=data['shape'],
                                    style=data['style'])
                        elif data['node_type'] == 'output':
                            layer.node(str(node), 
                                    f"{data['label']}\\n({data['connected_to']})",
                                    fillcolor=data['color'],
                                    shape=data['shape'],
                                    style=data['style'])
        
        # Adiciona arestas
        for source, target, data in self.G.edges(data=True):
            dot.edge(str(source), str(target),
                    color=data['color'],
                    penwidth=str(data['penwidth']),
                    style=data.get('style', 'solid'))
        
        # Salva em diferentes formatos
        dot.render(save_path, format='png', cleanup=True)
        dot.render(save_path, format='svg', cleanup=True)
        
        print(f"Diagrama hierárquico salvo em: {save_path}.png, {save_path}.svg")
        return dot
    
    def create_force_directed_diagram(self, save_path: str = "cgp_force_directed"):
        """Cria diagrama com layout de força dirigida"""
        parsed_genome = self.parse_genome()
        active_nodes = self.find_active_nodes(parsed_genome)
        
        # Constrói o grafo
        self.build_network_graph(parsed_genome, active_nodes)
        
        # Cria o grafo Graphviz com layout de força
        dot = graphviz.Digraph(comment='CGP Force Directed Diagram')
        dot.attr(layout='neato', size='20,14', dpi='300', overlap='false', splines='true')
        dot.attr('node', fontsize='10', fontname='Arial')
        dot.attr('edge', fontsize='8', fontname='Arial')
        
        # Adiciona nós
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
        
        # Adiciona arestas
        for source, target, data in self.G.edges(data=True):
            dot.edge(str(source), str(target),
                    color=data['color'],
                    penwidth=str(data['penwidth']),
                    style=data.get('style', 'solid'))
        
        # Salva em diferentes formatos
        dot.render(save_path, format='png', cleanup=True)
        dot.render(save_path, format='svg', cleanup=True)
        
        print(f"Diagrama força dirigida salvo em: {save_path}.png, {save_path}.svg")
        return dot
    
    def create_circular_diagram(self, save_path: str = "cgp_circular"):
        """Cria diagrama circular"""
        parsed_genome = self.parse_genome()
        active_nodes = self.find_active_nodes(parsed_genome)
        
        # Constrói o grafo
        self.build_network_graph(parsed_genome, active_nodes)
        
        # Cria o grafo Graphviz circular
        dot = graphviz.Digraph(comment='CGP Circular Diagram')
        dot.attr(layout='circo', size='20,14', dpi='300')
        dot.attr('node', fontsize='10', fontname='Arial')
        dot.attr('edge', fontsize='8', fontname='Arial')
        
        # Adiciona nós
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
        
        # Adiciona arestas
        for source, target, data in self.G.edges(data=True):
            dot.edge(str(source), str(target),
                    color=data['color'],
                    penwidth=str(data['penwidth']),
                    style=data.get('style', 'solid'))
        
        # Salva em diferentes formatos
        dot.render(save_path, format='png', cleanup=True)
        dot.render(save_path, format='svg', cleanup=True)
        
        print(f"Diagrama circular salvo em: {save_path}.png, {save_path}.svg")
        return dot
    
    def _organize_nodes_in_layers(self):
        """Organiza nós em camadas hierárquicas"""
        layers = []
        processed = set()
        
        # Encontra inputs utilizados
        used_inputs = set()
        for node in self.G.nodes():
            if self.G.nodes[node]['node_type'] == 'input':
                used_inputs.add(node)
        
        # Camada 0: inputs
        layers.append(list(used_inputs))
        processed.update(used_inputs)
        
        current_layer = list(used_inputs)
        
        while current_layer and len(layers) < 5:  # Máximo 5 camadas
            next_layer = []
            for node_id in current_layer:
                for successor in self.G.successors(node_id):
                    if successor not in processed:
                        if self.G.nodes[successor]['node_type'] == 'processing':
                            next_layer.append(successor)
                            processed.add(successor)
            
            if next_layer:
                layers.append(next_layer)
                current_layer = next_layer
            else:
                break
        
        # Adiciona outputs na última camada
        output_nodes = [node for node in self.G.nodes() 
                       if self.G.nodes[node]['node_type'] == 'output']
        if output_nodes:
            layers.append(output_nodes)
        
        return layers
    
    def create_all_layouts(self):
        """Cria todos os tipos de layout"""
        print("Criando diagramas com diferentes layouts do Graphviz...")
        
        # Layout hierárquico
        print("\n1. Layout Hierárquico (dot)")
        self.create_hierarchical_diagram()
        
        # Layout força dirigida
        print("\n2. Layout Força Dirigida (neato)")
        self.create_force_directed_diagram()
        
        # Layout circular
        print("\n3. Layout Circular (circo)")
        self.create_circular_diagram()
        
        print("\nTodos os diagramas foram criados com sucesso!")
    
    def get_network_info(self):
        """Retorna informações sobre a rede"""
        return {
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(),
            'input_nodes': len([n for n, d in self.G.nodes(data=True) if d['node_type'] == 'input']),
            'processing_nodes': len([n for n, d in self.G.nodes(data=True) if d['node_type'] == 'processing']),
            'output_nodes': len([n for n, d in self.G.nodes(data=True) if d['node_type'] == 'output'])
        }


def main():
    """Função principal"""
    if len(sys.argv) > 1:
        genome_file = sys.argv[1]
    else:
        genome_file = "best_genome.json"
    
    diagram = CGPAdvancedGraphviz(genome_file)
    
    # Cria todos os tipos de layout
    diagram.create_all_layouts()
    
    # Mostra informações da rede
    info = diagram.get_network_info()
    print(f"\nInformações da rede:")
    print(f"- Total de nós: {info['nodes']}")
    print(f"- Total de arestas: {info['edges']}")
    print(f"- Nós de entrada: {info['input_nodes']}")
    print(f"- Nós de processamento: {info['processing_nodes']}")
    print(f"- Nós de saída: {info['output_nodes']}")


if __name__ == "__main__":
    main()
