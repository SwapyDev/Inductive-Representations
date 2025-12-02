import torch
from torch_geometric.nn import GraphSAGE
from torch_geometric.datasets import Reddit
from train_reddit import SimpleRedditTrainer

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load Reddit
    print("Loading Reddit dataset (this may take a minute)")
    dataset = Reddit(root='data/Reddit')
    data = dataset[0]
    
    print("=" * 60)
    print("REDDIT DATASET")
    print("=" * 60)
    print(f"Nodes: {data.num_nodes:,}")
    print(f"Edges: {data.num_edges:,}")
    print(f"Features: {data.num_features}")
    print(f"Classes: {dataset.num_classes}")
    print(f"Train nodes: {data.train_mask.sum():,}")
    print(f"Val nodes: {data.val_mask.sum():,}")
    print(f"Test nodes: {data.test_mask.sum():,}")
    print("=" * 60)
    
    # Create model
    print("\nCreating GraphSAGE model...")
    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=256,
        num_layers=2,
        out_channels=dataset.num_classes,
        dropout=0.5,
        aggr='mean'
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nTraining")
    trainer = SimpleRedditTrainer(model, data, device=device, lr=0.01)
    results = trainer.train(epochs=100, patience=20, verbose=True)
    
    # Compare to paper
    print("\n" + "=" * 60)
    print("COMPARISON TO PAPER")
    print("=" * 60)
    print(f"Paper (GraphSAGE-mean supervised): 0.950")
    print(f"my result (F1 micro): {results['f1_micro']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()