from torch_geometric.datasets import PPI
from torch_geometric.utils import degree
#Use the PPI dataset from PyTorch Geometric

train = PPI(root='data/PPI', split='train')
val   = PPI(root='data/PPI', split='val')
test  = PPI(root='data/PPI', split='test')


print("PPI Dataset Info")
print(f"Train graphs: {len(train)}")
print(f"Val graphs:   {len(val)}")
print(f"Test graphs:  {len(test)}")

sample = train[0]

print("\n=== Single Graph Info ===")
print(f"Nodes: {sample.num_nodes}")
print(f"Edges: {sample.num_edges}")
print(f"Node features: {sample.num_features}")
print(f"Node labels (multi-label dim): {sample.y.size(1)}")

# Additional details
print("\n=== Edge Index Shape ===")
print(sample.edge_index.shape)

# Check average node degree
deg = degree(sample.edge_index[0], sample.num_nodes)
print(f"\nAverage degree: {deg.mean():.2f}")
print(f"Max degree:     {deg.max().item()}")
print(f"Min degree:     {deg.min().item()}")