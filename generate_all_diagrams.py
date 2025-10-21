#!/usr/bin/env python3
"""
Script unificado para gerar todos os tipos de diagramas CGP organizados na pasta 'diagramas'
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


class CGPAllDiagrams:
    def __init__(self, genome_file: str = "best_genome.json", output_dir: str = "diagramas"):
        """Inicializa o gerador de todos os diagramas"""
        self.genome_file = genome_file
        self.output_dir = output_dir
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
        
        # Criar diretório de saída
        os.makedirs(self.output_dir, exist_ok=True)
        
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
    
    def create_matplotlib_diagram(self):
        """Cria diagrama usando matplotlib (versão original melhorada)"""
        print("Gerando diagrama matplotlib...")
        
        parsed_genome = self.parse_genome()
        active_nodes = self.find_active_nodes(parsed_genome)
        
        # Constrói o grafo
        self.build_network_graph(parsed_genome, active_nodes)
        
        # Organiza nós em camadas
        layers = self._organize_nodes_in_layers()
        
        fig, ax = plt.subplots(figsize=(20, 14))
        
        # Desenha o diagrama
        self._draw_circuit_layout(ax, parsed_genome, active_nodes, layers)
        
        # Configurações do plot
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 14)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.title("CGP Circuit Diagram - Matplotlib Version", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, "cgp_matplotlib_diagram.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Salvo em: {save_path}")
    
    def create_graphviz_hierarchical(self):
        """Cria diagrama hierárquico usando Graphviz"""
        print("Gerando diagrama Graphviz hierárquico...")
        
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
        base_path = os.path.join(self.output_dir, "cgp_hierarchical")
        dot.render(base_path, format='png', cleanup=True)
        dot.render(base_path, format='svg', cleanup=True)
        dot.render(base_path, format='pdf', cleanup=True)
        
        print(f"  ✓ Salvo em: {base_path}.png, {base_path}.svg, {base_path}.pdf")
    
    def create_graphviz_force_directed(self):
        """Cria diagrama com layout de força dirigida"""
        print("Gerando diagrama Graphviz força dirigida...")
        
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
        base_path = os.path.join(self.output_dir, "cgp_force_directed")
        dot.render(base_path, format='png', cleanup=True)
        dot.render(base_path, format='svg', cleanup=True)
        dot.render(base_path, format='pdf', cleanup=True)
        
        print(f"  ✓ Salvo em: {base_path}.png, {base_path}.svg, {base_path}.pdf")
    
    def create_graphviz_circular(self):
        """Cria diagrama circular"""
        print("Gerando diagrama Graphviz circular...")
        
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
        base_path = os.path.join(self.output_dir, "cgp_circular")
        dot.render(base_path, format='png', cleanup=True)
        dot.render(base_path, format='svg', cleanup=True)
        dot.render(base_path, format='pdf', cleanup=True)
        
        print(f"  ✓ Salvo em: {base_path}.png, {base_path}.svg, {base_path}.pdf")
    
    def create_networkx_spring_layout(self):
        """Cria diagrama usando NetworkX com layout spring"""
        print("Gerando diagrama NetworkX spring layout...")
        
        parsed_genome = self.parse_genome()
        active_nodes = self.find_active_nodes(parsed_genome)
        
        # Constrói o grafo
        self.build_network_graph(parsed_genome, active_nodes)
        
        # Calcula layout spring
        pos = nx.spring_layout(self.G, k=3, iterations=50, seed=42)
        
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
        
        plt.title("CGP Network Diagram - NetworkX Spring Layout", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, "cgp_networkx_spring.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Salvo em: {save_path}")
    
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
    
    def _draw_circuit_layout(self, ax, parsed_genome, active_nodes, layers):
        """Desenha o layout do circuito (versão matplotlib)"""
        # Desenha nós de entrada
        self._draw_input_nodes(ax, layers[0])
        
        # Desenha nós de processamento
        processing_layers = layers[1:] if len(layers) > 1 else []
        for i, layer in enumerate(processing_layers):
            self._draw_processing_layer(ax, layer, parsed_genome, i + 1)
        
        # Desenha nós de saída
        self._draw_output_nodes(ax, parsed_genome)
        
        # Desenha conexões
        self._draw_connections(ax, parsed_genome, layers)
    
    def _draw_input_nodes(self, ax, input_nodes):
        """Desenha nós de entrada"""
        for i, node_id in enumerate(input_nodes):
            x = 1
            y = 10 - i * 2
            
            # Círculo do nó de entrada
            circle = Circle((x, y), 0.3, color='black', fill=False, linewidth=2)
            ax.add_patch(circle)
            
            # Label do nó
            ax.text(x, y, f'x{node_id}', ha='center', va='center', 
                   fontsize=10, fontweight='bold')
            
            # Porta de saída
            output_circle = Circle((x + 0.4, y), 0.12, color='black', fill=True)
            ax.add_patch(output_circle)
            ax.text(x + 0.4, y, str(node_id), ha='center', va='center', 
                   fontsize=10, color='white', fontweight='bold')
    
    def _draw_processing_layer(self, ax, layer_nodes, parsed_genome, layer_idx):
        """Desenha uma camada de nós de processamento"""
        max_nodes_per_row = 6
        rows = (len(layer_nodes) + max_nodes_per_row - 1) // max_nodes_per_row
        
        for i, node_id in enumerate(layer_nodes):
            if node_id in parsed_genome['nodes']:
                node_data = parsed_genome['nodes'][node_id]
                row = i // max_nodes_per_row
                col = i % max_nodes_per_row
                
                x = 4 + col * 2.5
                y = 11 - row * 2.5
                
                self._draw_processing_node(ax, x, y, node_id, node_data)
    
    def _draw_processing_node(self, ax, x, y, node_id, node_data):
        """Desenha um nó de processamento"""
        # Retângulo com bordas arredondadas
        rect = FancyBboxPatch((x - 0.6, y - 0.4), 1.2, 0.8,
                             boxstyle="round,pad=0.05",
                             facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Nome da função
        function_name = node_data['function_name']
        ax.text(x, y, function_name, ha='center', va='center', 
               fontsize=12, fontweight='bold', color='magenta')
        
        # Portas de entrada
        input1_circle = Circle((x - 0.6, y - 0.25), 0.1, color='black', fill=True)
        ax.add_patch(input1_circle)
        ax.text(x - 0.6, y - 0.25, str(node_data['input1']), ha='center', va='center', 
               fontsize=9, color='white', fontweight='bold')
        
        if node_data['n_inputs'] == 2:
            input2_circle = Circle((x - 0.6, y + 0.25), 0.1, color='black', fill=True)
            ax.add_patch(input2_circle)
            ax.text(x - 0.6, y + 0.25, str(node_data['input2']), ha='center', va='center', 
                   fontsize=9, color='white', fontweight='bold')
        
        # Porta de saída
        output_circle = Circle((x + 0.6, y), 0.1, color='black', fill=True)
        ax.add_patch(output_circle)
        ax.text(x + 0.6, y, str(node_id), ha='center', va='center', 
               fontsize=9, color='white', fontweight='bold')
        
        # ID do nó
        ax.text(x + 0.4, y - 0.3, str(node_id), ha='center', va='center', 
               fontsize=10, color='red', fontweight='bold')
    
    def _draw_output_nodes(self, ax, parsed_genome):
        """Desenha nós de saída"""
        active_nodes = self.find_active_nodes(parsed_genome)
        connected_outputs = []
        
        for i, output_node in enumerate(parsed_genome['outputs']):
            if output_node in active_nodes or output_node < 3:
                connected_outputs.append((i, output_node))
        
        for i, (output_idx, output_node) in enumerate(connected_outputs):
            x = 18
            y = 11 - i * 1.5
            
            # Círculo do nó de saída
            circle = Circle((x, y), 0.4, color='black', fill=False, linewidth=2)
            ax.add_patch(circle)
            
            # Porta de entrada
            input_circle = Circle((x - 0.5, y), 0.12, color='black', fill=True)
            ax.add_patch(input_circle)
            ax.text(x - 0.5, y, str(output_node), ha='center', va='center', 
                   fontsize=10, color='white', fontweight='bold')
    
    def _draw_connections(self, ax, parsed_genome, layers):
        """Desenha conexões entre nós"""
        # Conexões dos inputs para nós de processamento
        for layer_idx, layer in enumerate(layers[1:], 1):
            for node_id in layer:
                if node_id in parsed_genome['nodes']:
                    node_data = parsed_genome['nodes'][node_id]
                    self._draw_node_connections(ax, node_id, node_data, parsed_genome)
        
        # Conexões para outputs
        for i in range(min(8, len(parsed_genome['outputs']))):
            output_node = parsed_genome['outputs'][i]
            self._draw_output_connection(ax, output_node, i)
    
    def _draw_node_connections(self, ax, node_id, node_data, parsed_genome):
        """Desenha conexões de um nó específico"""
        # Implementação simplificada para conexões
        pass
    
    def _draw_output_connection(self, ax, output_node, output_idx):
        """Desenha conexão para um nó de saída"""
        # Implementação simplificada para conexões
        pass
    
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
        circle = Circle((x, y), 0.3, color='black', fill=True, 
                       facecolor=data['color'], linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, data['label'], ha='center', va='center', 
               fontsize=10, fontweight='bold')
    
    def _draw_processing_node_networkx(self, ax, x, y, node, data):
        """Desenha nó de processamento no NetworkX"""
        rect = FancyBboxPatch((x - 0.4, y - 0.2), 0.8, 0.4,
                             boxstyle="round,pad=0.05",
                             facecolor=data['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, data['function_name'], ha='center', va='center', 
               fontsize=9, fontweight='bold')
    
    def _draw_output_node_networkx(self, ax, x, y, node, data):
        """Desenha nó de saída no NetworkX"""
        outer_circle = Circle((x, y), 0.3, color='black', fill=False, linewidth=3)
        inner_circle = Circle((x, y), 0.2, color='black', fill=True, 
                             facecolor=data['color'], linewidth=2)
        ax.add_patch(outer_circle)
        ax.add_patch(inner_circle)
        ax.text(x, y, data['label'], ha='center', va='center', 
               fontsize=10, fontweight='bold')
    
    def create_network_analysis(self):
        """Cria análise da rede e salva na pasta diagramas"""
        print("Gerando análise da rede...")
        
        parsed_genome = self.parse_genome()
        active_nodes = self.find_active_nodes(parsed_genome)
        self.build_network_graph(parsed_genome, active_nodes)
        
        info = {
            'total_nodes': self.G.number_of_nodes(),
            'total_edges': self.G.number_of_edges(),
            'input_nodes': len([n for n, d in self.G.nodes(data=True) if d['node_type'] == 'input']),
            'processing_nodes': len([n for n, d in self.G.nodes(data=True) if d['node_type'] == 'processing']),
            'output_nodes': len([n for n, d in self.G.nodes(data=True) if d['node_type'] == 'output']),
            'functions_used': list(set([d['function_name'] for n, d in self.G.nodes(data=True) 
                                      if d['node_type'] == 'processing']))
        }
        
        # Salva análise
        analysis_path = os.path.join(self.output_dir, "network_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"  ✓ Análise salva em: {analysis_path}")
        
        # Cria relatório em texto
        report_path = os.path.join(self.output_dir, "network_report.txt")
        with open(report_path, 'w') as f:
            f.write("=== CGP Network Analysis Report ===\n\n")
            f.write(f"Total de nós: {info['total_nodes']}\n")
            f.write(f"Total de arestas: {info['total_edges']}\n")
            f.write(f"Nós de entrada: {info['input_nodes']}\n")
            f.write(f"Nós de processamento: {info['processing_nodes']}\n")
            f.write(f"Nós de saída: {info['output_nodes']}\n\n")
            f.write("Funções utilizadas:\n")
            for func in sorted(info['functions_used']):
                f.write(f"  - {func}\n")
        
        print(f"  ✓ Relatório salvo em: {report_path}")
    
    def generate_all_diagrams(self):
        """Gera todos os tipos de diagramas"""
        print("=== Gerando todos os diagramas CGP ===\n")
        print(f"Pasta de saída: {self.output_dir}/")
        print()
        
        # 1. Diagrama matplotlib
        self.create_matplotlib_diagram()
        print()
        
        # 2. Diagramas Graphviz
        self.create_graphviz_hierarchical()
        print()
        self.create_graphviz_force_directed()
        print()
        self.create_graphviz_circular()
        print()
        
        # 3. Diagrama NetworkX
        self.create_networkx_spring_layout()
        print()
        
        # 4. Análise da rede
        self.create_network_analysis()
        print()
        
        print("=== Todos os diagramas foram gerados com sucesso! ===")
        print(f"Verifique a pasta '{self.output_dir}/' para ver todos os arquivos.")


def main():
    """Função principal"""
    if len(sys.argv) > 1:
        genome_file = sys.argv[1]
    else:
        genome_file = "best_genome.json"
    
    # Gera todos os diagramas
    diagram_generator = CGPAllDiagrams(genome_file)
    diagram_generator.generate_all_diagrams()


if __name__ == "__main__":
    main()
