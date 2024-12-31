# This is LLM queried with : 
#  using networkx create me a triangular lattice graph, a square lattice graph,
#  and an hexagonal lattice graph
#  display them with a script in order to check whether you did good or not
#

##
# To get the neighbors of one of the vertex of the graph do
# neighbors = G.neighbors(node)
##

import networkx as nx
import matplotlib.pyplot as plt
import math
import random


def pick_two_random_vertices(G):
    vertices = list(G.nodes())
    v1, v2 = random.sample(vertices, 2)
    return v1, v2


# Create a triangular lattice graph
def create_triangular_lattice(rows, cols):
    G = nx.triangular_lattice_graph(rows, cols, periodic=True)
    mapping = {node: idx for idx, node in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, mapping)
    return G

# Create a square lattice graph
def create_square_lattice(rows, cols):
    G = nx.grid_2d_graph(rows, cols, periodic=True)
    mapping = {node: idx for idx, node in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, mapping)
    return G

# Create a hexagonal lattice graph
def create_hexagonal_lattice(rows, cols):
    G = nx.hexagonal_lattice_graph(rows, cols, periodic=True)
    mapping = {node: idx for idx, node in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, mapping)
    return G


# Plot function to make sure output of LLM is the awaited logic structure
def plot_graph(G, title):
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=False, node_size=30, node_color="lightblue", edge_color="gray")
    plt.title(title)
    plt.axis("equal")
    plt.show()

# Main script
if __name__ == "__main__":
    # Triangular lattice
    tri_lattice = create_triangular_lattice(5, 5)
    plot_graph(tri_lattice, "Triangular Lattice")

    # Square lattice
    square_lattice = create_square_lattice(5, 5)
    plot_graph(square_lattice, "Square Lattice")

    # Hexagonal lattice
    hex_lattice = create_hexagonal_lattice(6, 6)
    plot_graph(hex_lattice, "Hexagonal Lattice")

    G = nx.hexagonal_lattice_graph(10,10, periodic=True)
    print("number of nodes of G = ", G.number_of_nodes())
    mapping = {node: idx for idx, node in enumerate(G.nodes)}
    
    # Relabel nodes in the graph using the mapping
    G = nx.relabel_nodes(G, mapping)
    print(list(G.neighbors(1)))
    import random
    random_node = random.randint(0, G.number_of_nodes() - 1)
    print("Random node:", random_node, "Neighbors:", list(G.neighbors(random_node)))