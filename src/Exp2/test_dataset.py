from torch_geometric.datasets import PPI
#this is just to verify that we are using the same dataset as the paper!

train_dataset = PPI(root='data/PPI', split='train')
val_dataset = PPI(root='data/PPI', split='val')
test_dataset = PPI(root='data/PPI', split='test')

print("Verificación del dataset:")
print(f"Train graphs: {len(train_dataset)}")  
print(f"Val graphs: {len(val_dataset)}")      
print(f"Test graphs: {len(test_dataset)}")  
print(f"Node features: {train_dataset[0].num_features}")  
print(f"Classes: {train_dataset.num_classes}")  

# Verify graph size
print("\nSample graph sizes:")
for i in range(3):
    g = train_dataset[i]
    print(f"Graph {i}: {g.num_nodes} nodes, {g.num_edges} edges")