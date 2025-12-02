import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import PPI
from torch_geometric.utils import degree
from torch_geometric.nn import GraphSAGE

# Import your  trainers
from train import Trainer  # supervised trainer
from train_unsupervised import UnsupervisedTrainer # unsupervised trainer

class DistributionShiftLoader:
    def __init__(self, root="data/PPI"):
        self.root = root

    def load_ood_split(self, split_ratio=0.7):
        print("Loading and analyzing graph structures")
        # Pool all data
        dataset = []
        for split in ['train', 'val', 'test']:
            dataset.extend(list(PPI(root=self.root, split=split)))
        
        # Calculate Average Degree for each graph
        graph_data = []
        for i, data in enumerate(dataset):
            d = degree(data.edge_index[0], data.num_nodes)
            graph_data.append({
                'data': data,
                'avg_degree': d.mean().item()
            })
        
        # Sort by density (Low -> High)
        graph_data.sort(key=lambda x: x['avg_degree'])
        
        # Split
        split_idx = int(len(graph_data) * split_ratio)
        train_val_set = graph_data[:split_idx]  # The Sparse ones
        test_set = graph_data[split_idx:]       # The Dense ones
        
        # Create validation set from the Training distribution
        val_idx = int(len(train_val_set) * 0.8)
        
        train_graphs = [x['data'] for x in train_val_set[:val_idx]]
        val_graphs   = [x['data'] for x in train_val_set[val_idx:]]
        test_graphs  = [x['data'] for x in test_set]
        
        # Stats
        train_deg = np.mean([x['avg_degree'] for x in train_val_set[:val_idx]])
        test_deg = np.mean([x['avg_degree'] for x in test_set])
        
        print(f"Train (Sparse): {len(train_graphs)} graphs | Avg Degree: {train_deg:.2f}")
        print(f"Test  (Dense):  {len(test_graphs)} graphs | Avg Degree: {test_deg:.2f}")
        print(f"Shift Magnitude: +{test_deg - train_deg:.2f} avg neighbors")
        
        return train_graphs, val_graphs, test_graphs

# --- 2. Experiment Runners ---

def run_supervised(train_loader, val_loader, test_loader, device, aggregator):
    print(f"\n[SUPERVISED] Training on Sparse -> Testing on Dense ({aggregator.upper()})")
    
    in_channels = train_loader.dataset[0].num_features
    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=256, # Paper default
        num_layers=2,
        out_channels=121,    # Classes
        aggr=aggregator
    ).to(device)
    
    trainer = Trainer(model, device=device, lr=0.005)
    trainer.train(train_loader, val_loader, epochs=50, patience=10, verbose=False)
    
    metrics = trainer.evaluate(test_loader)
    print(f"   -> Result: Micro-F1 = {metrics['micro_f1']:.4f}")
    return metrics['micro_f1']

def run_unsupervised(train_loader, val_loader, test_loader, device, aggregator):
    print(f"\n[UNSUPERVISED] Training on Sparse -> Testing on Dense ({aggregator.upper()})")
    
    in_channels = train_loader.dataset[0].num_features
    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=256,
        num_layers=2,
        out_channels=256,    # Embeddings
        aggr=aggregator
    ).to(device)
    
    trainer = UnsupervisedTrainer(model, device=device, lr=0.00001) # Lower LR for unsup usually better
    
    # Train Embeddings (Unsupervised)
    # Note: We pass 'verbose=False' to avoid spamming logs
    classifiers = trainer.train(train_loader, val_loader, epochs=10, verbose=False)
    
    # Evaluate Classifier on Test Embeddings
    test_embeddings, test_labels = trainer.generate_embeddings(test_loader)
    metrics = trainer.evaluate_with_classifier(classifiers, test_embeddings, test_labels)
    
    print(f"   -> Result: Micro-F1 = {metrics['micro_f1']:.4f}")
    return metrics['micro_f1']

#  Main Execution 

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load OOD Data
    loader = DistributionShiftLoader()
    train_graphs, val_graphs, test_graphs = loader.load_ood_split()
    
    # Create DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=2, shuffle=True)
    val_loader   = DataLoader(val_graphs, batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_graphs, batch_size=1, shuffle=False)
    
    results = []
    
    # Run Experiments
    aggregators = ['mean', 'max'] 
    
    for aggr in aggregators:
        #Supervised
        sup_score = run_supervised(train_loader, val_loader, test_loader, device, aggr)
        
        #Unsupervised
        unsup_score = run_unsupervised(train_loader, val_loader, test_loader, device, aggr)
        
        results.append({
            'Aggregator': aggr.upper(),
            'Supervised F1': sup_score,
            'Unsupervised F1': unsup_score,
            'Gap': sup_score - unsup_score
        })
        
    # Print Final Summary
    print("\n" + "="*60)
    print("FINAL RESULTS: STRUCTURAL GENERALIZATION")
    print("="*60)
    df = pd.DataFrame(results)
    print(df.round(4))
    print("="*60)