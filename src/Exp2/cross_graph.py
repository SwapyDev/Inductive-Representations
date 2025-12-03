"""
Cross-Graph Generalization Experiment
Train on subset of PPI graphs, test on different ones
"""

import torch
from torch_geometric.nn import GraphSAGE
from dataloader import DataLoader as PPIDataLoader
from train import train_and_evaluate
from torch_geometric.loader import DataLoader
from train import Trainer
import pandas as pd
import numpy as np


def split_graphs_by_structure(dataset, split_ratio=0.5):
    """Split graphs by average degree (structural diversity)"""
    
    # Calculate average degree for each graph
    graph_stats = []
    for idx, data in enumerate(dataset):
        avg_degree = data.num_edges / data.num_nodes
        graph_stats.append((idx, avg_degree, data.num_nodes))
    
    # Sort by average degree
    graph_stats.sort(key=lambda x: x[1])
    
    # Split into low-degree and high-degree groups
    split_point = int(len(graph_stats) * split_ratio)
    
    low_degree_indices = [x[0] for x in graph_stats[:split_point]]
    high_degree_indices = [x[0] for x in graph_stats[split_point:]]
    
    return low_degree_indices, high_degree_indices


def cross_graph_experiment(aggregator="mean", device="cpu"):
    """
    Cross-graph generalization experiment.
    Train on low-degree graphs, test on high-degree graphs (and vice versa).
    """
    
    print("\n" + "=" * 70)
    print(f"CROSS-GRAPH GENERALIZATION - {aggregator.upper()}")
    print("=" * 70)
    
    # Load data
    print("\nLoading PPI dataset...")
    loader = PPIDataLoader(useStructuralFeatures=False)
    train_dataset, val_dataset, test_dataset = loader.load(verbose=False)
    
    # Split train graphs by structure
    low_degree_idx, high_degree_idx = split_graphs_by_structure(train_dataset)
    
    print(f"\nTrain set split:")
    print(f"  Low-degree graphs: {len(low_degree_idx)}")
    print(f"  High-degree graphs: {len(high_degree_idx)}")
    
    results = []
    
    # Experiment 1: Train on low-degree, test on high-degree
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Train on LOW-degree → Test on HIGH-degree")
    print("=" * 70)
    
    train_low = [train_dataset[i] for i in low_degree_idx]
    train_high = [train_dataset[i] for i in high_degree_idx]
    
    model = GraphSAGE(
        in_channels=50,
        hidden_channels=128,
        num_layers=2,
        out_channels=121,
        dropout=0.5,
        aggr=aggregator,
    )
    
    # Train on low-degree graphs
    train_loader = DataLoader(train_low, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    trainer = Trainer(model, device=device)
    best_val_f1 = trainer.train(
        train_loader, val_loader, epochs=100, patience=20, verbose=True
    )
    
    # Test on high-degree graphs (cross-graph)
    test_high_loader = DataLoader(train_high, batch_size=1, shuffle=False)
    test_metrics = trainer.evaluate(test_high_loader)
    
    # Test on original test set
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_original = trainer.evaluate(test_loader)
    
    print(f"\nResults:")
    print(f"  Val F1 (low-degree):          {best_val_f1:.4f}")
    print(f"  Test F1 (high-degree graphs): {test_metrics['micro_f1']:.4f}")
    print(f"  Test F1 (original test):      {test_original['micro_f1']:.4f}")
    
    results.append({
        'experiment': 'Low→High',
        'aggregator': aggregator.upper(),
        'train_on': 'Low-degree',
        'test_on': 'High-degree',
        'val_f1': best_val_f1,
        'cross_graph_f1': test_metrics['micro_f1'],
        'original_test_f1': test_original['micro_f1'],
    })
    
    # Experiment 2: Train on high-degree, test on low-degree
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Train on HIGH-degree → Test on LOW-degree")
    print("=" * 70)
    
    model = GraphSAGE(
        in_channels=50,
        hidden_channels=128,
        num_layers=2,
        out_channels=121,
        dropout=0.5,
        aggr=aggregator,
    )
    
    # Train on high-degree graphs
    train_loader = DataLoader(train_high, batch_size=4, shuffle=True)
    
    trainer = Trainer(model, device=device)
    best_val_f1 = trainer.train(
        train_loader, val_loader, epochs=100, patience=20, verbose=True
    )
    
    # Test on low-degree graphs (cross-graph)
    test_low_loader = DataLoader(train_low, batch_size=1, shuffle=False)
    test_metrics = trainer.evaluate(test_low_loader)
    
    # Test on original test set
    test_original = trainer.evaluate(test_loader)
    
    print(f"\nResults:")
    print(f"  Val F1 (high-degree):        {best_val_f1:.4f}")
    print(f"  Test F1 (low-degree graphs): {test_metrics['micro_f1']:.4f}")
    print(f"  Test F1 (original test):     {test_original['micro_f1']:.4f}")
    
    results.append({
        'experiment': 'High→Low',
        'aggregator': aggregator.upper(),
        'train_on': 'High-degree',
        'test_on': 'Low-degree',
        'val_f1': best_val_f1,
        'cross_graph_f1': test_metrics['micro_f1'],
        'original_test_f1': test_original['micro_f1'],
    })
    
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    aggregators = ["mean", "max"]
    
    all_results = []
    
    for aggr in aggregators:
        results = cross_graph_experiment(aggregator=aggr, device=device)
        all_results.extend(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("CROSS-GRAPH GENERALIZATION RESULTS")
    print("=" * 70)
    
    df = pd.DataFrame(all_results)
    print("\n" + df.to_string(index=False))
    
    # Analysis
    print("\n" + "=" * 70)
    print("GENERALIZATION GAP ANALYSIS")
    print("=" * 70)
    
    for _, row in df.iterrows():
        gap = row['original_test_f1'] - row['cross_graph_f1']
        gap_pct = (gap / row['original_test_f1']) * 100
        
        print(f"\n{row['aggregator']} - {row['experiment']}:")
        print(f"  Original test F1:    {row['original_test_f1']:.4f}")
        print(f"  Cross-graph F1:      {row['cross_graph_f1']:.4f}")
        print(f"  Generalization gap:  {gap:.4f} ({gap_pct:.1f}%)")
    
    # Save
    df.to_csv("cross_graph_results.csv", index=False)
    print("\n✓ Results saved to cross_graph_results.csv")


if __name__ == "__main__":
    main()