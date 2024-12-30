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

# Create a triangular lattice graph
def create_triangular_lattice(rows, cols):
    G = nx.Graph()
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            G.add_node(idx, pos=(col, -row * math.sqrt(3) / 2))
            
            # Connect right neighbor
            if col < cols - 1:
                G.add_edge(idx, idx + 1)
            # Connect bottom neighbor
            if row < rows - 1:
                G.add_edge(idx, idx + cols)
            # Connect diagonal neighbor
            if row < rows - 1 and col < cols - 1:
                G.add_edge(idx, idx + cols + 1)
    return G

# Create a square lattice graph
def create_square_lattice(rows, cols):
    G = nx.grid_2d_graph(rows, cols)
    pos = {n: (n[1], -n[0]) for n in G.nodes()}  # Arrange grid coordinates 
    nx.set_node_attributes(G, pos, 'pos')           #  only for visualisation purposes
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

