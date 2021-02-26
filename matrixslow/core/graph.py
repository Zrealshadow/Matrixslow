'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2021-02-23 11:23:14
 * @desc 
'''

import os
SEED = 2020

class Graph(object):
    """Computing Graph Class"""

    def __init__(self):
        self.nodes = []
        self.name_scope = None

        self.node_size=2000
        self.edgecolors = "#666666"
        self.node_color = "#999999"
        self.edge_color = "#014b66"
        self.edge_width = 2
        self.font_weight = "bold"
        self.font_color = "#6c6c6c"
        self.font_size = 8
        self.font_family = 'arial'
        # self.
    # from .node import Node
    def add_node(self, node) -> None:
        """Add Node into Computing Graph
        """
        self.nodes.append(node)

    
    def clear_jacobi(self) -> None:
        """clear all jacobi matrix of nodes
        """
        for node in self.nodes:
            node.clear_jacobi()
    
    def node_count(self) -> int:
        """Count the nodes in computing graph"""
        return len(self.nodes)
    
    def draw(self, ax=None, filepath:str = None):
        """plot the whole computing graph"""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            import numpy as np
        except:
            raise Exception("Need External Module : networkx , matplotlib")
        G = nx.Graph()
        already_edge = []
        labels = {}

        for node in self.nodes:
            G.add_node(node)
            labels[node] = (node.__class__.__name__ + 
                ("({:s})".format(str(node.dim)) if hasattr(node, "dim") else "") +
                ("\n[{:.3f}]".format(np.linalg.norm(node.jacobi)) if node.jacobi is not None else "") 
                )

            for c in node.get_children():
                if {node, c} not in already_edge:
                    G.add_edge(node, c)
                    already_edge.append({node, c})
            
            for p in node.get_parents():
                if {p, node} not in already_edge:
                    G.add_edge(p,node)
                    already_edge.append({p, node})

        if ax is None:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)
        
        ax.clear()
        ax.axis("on")
        ax.grid(True)

        pos = nx.spring_layout(G, seed = SEED)
        #different node using different color 
        variable_jacobi_list = []
        variable_no_jacobi_list = []
        node_jacobi_list = []
        node_no_jacobi_list = []

        for n in self.nodes:
            if n.__class__.__name__ == "Variable" and n.jacobi is not None:
                variable_jacobi_list.append(n)
            elif n.__class__.__name__ == "Variable":
                variable_no_jacobi_list.append(n)
            elif n.jacobi is not None:
                node_jacobi_list.append(n)
            else:
                node_no_jacobi_list.append(n)

        cm = plt.cm.Reds #colormap
        colorlist = [np.linalg.norm(n.jacobi) for n in variable_jacobi_list]
        nx.draw_networkx_nodes(G, pos=pos, ax=ax, nodelist=variable_jacobi_list, node_color=colorlist, cmap=cm, edgecolors=self.edgecolors, node_size=self.node_size, alpha=1.0)


        nx.draw_networkx_nodes(G, pos=pos, ax=ax, nodelist=variable_no_jacobi_list, node_color = self.node_color, cmap=cm, edgecolors=self.edgecolors, node_size=self.node_size, alpha=1.0)

        colorlist = [np.linalg.norm(n.jacobi) for n in node_jacobi_list]
        nx.draw_networkx_nodes(G, pos=pos, ax=ax, nodelist=node_jacobi_list, node_color=colorlist, cmap=cm, edgecolors=self.edgecolors, node_size=self.node_size, alpha=1.0)

        nx.draw_networkx_nodes(G, pos=pos, ax=ax, nodelist=node_no_jacobi_list, node_color= self.node_color, cmap=cm, edgecolors=self.edgecolors, node_size=self.node_size, alpha=1.0)
    
        nx.draw_networkx_edges(G, pos=pos, ax=ax, width=self.edge_width, edge_color=self.edge_color)
        
        nx.draw_networkx_labels(G, pos=pos, ax=ax,labels=labels, font_weight=self.font_weight, font_color=self.font_color, font_size=self.font_size, font_family=self.font_family)

        if filepath is not None:
            filename = "fig"
            p = os.path.join(filepath,filename)
            i = 1
            while os.path.exists(p):
                filename = "fig"+ str(i)
                p = os.path.join(filepath, filename)
                i += 1
            plt.savefig(p)

#default graph for computing
default_graph = Graph()