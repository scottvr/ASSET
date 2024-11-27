import networkx as nx
from graphscii import ASCIIGraphRenderer

# Create a simple graph
G = nx.DiGraph()
G.add_edges_from([("A", "B"), ("A", "C"), ("B", "D")])

# Render it in ASCII
renderer = ASCIIGraphRenderer(G)
print(renderer.render())
