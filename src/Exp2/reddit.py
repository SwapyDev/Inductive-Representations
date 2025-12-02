from torch_geometric.datasets import Reddit

# Load Reddit dataset
dataset = Reddit(root="data/Reddit")
data = dataset[0]  # Single large graph

print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.num_edges}")
print(f"Features: {data.num_features}")
print(f"Classes: {dataset.num_classes}")
