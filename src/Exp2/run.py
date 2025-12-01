import torch
from torch_geometric.nn import GraphSAGE
from dataloader import DataLoader as PPIDataLoader
from train import train_and_evaluate
import pandas as pd

def run_single_experiment(use_structural_features, aggregator, device='cpu', epochs=100, patience=20):
    """Run a single experiment."""
    
    exp_name = f"{'Augmented' if use_structural_features else 'Baseline'}-{aggregator.upper()}"
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 70)
    
    # Load data
    print("\nLoading dataset...")
    data_loader = PPIDataLoader(useStructuralFeatures=use_structural_features)
    train_dataset, val_dataset, test_dataset = data_loader.load(verbose=False)
    
    in_channels = train_dataset[0].num_features
    print(f"Features: {in_channels}")
    
    # Create model
    print(f"Creating {aggregator.upper()} model...")
    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=256,
        num_layers=3,
        out_channels=121,
        dropout=0.5,
        aggr=aggregator
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    # Train
    results = train_and_evaluate(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device,
        epochs=epochs,
        patience=patience,
        verbose=True
    )
    
    print(f"\n✓ {exp_name} complete!")
    print(f"  Best Val F1: {results['best_val_f1']:.4f}")
    print(f"  Test Micro-F1: {results['test_micro_f1']:.4f}")
    
    return {
        'experiment': exp_name,
        'features': 'Augmented' if use_structural_features else 'Baseline',
        'aggregator': aggregator.upper(),
        'input_dim': in_channels,
        'parameters': num_params,
        'val_f1': results['best_val_f1'],
        'test_micro_f1': results['test_micro_f1'],
        'test_macro_f1': results['test_macro_f1']
    }


def main():
    """Run full experiment suite."""
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration
    aggregators = ['mean', 'max', 'lstm']  # Note: built-in uses 'max' instead of 'pool'
    epochs = 100
    patience = 20
    
    print("\n" + "=" * 70)
    print("FULL EXPERIMENT: Structural Feature Augmentation Study")
    print("=" * 70)
    print(f"Aggregators: {aggregators}")
    print(f"Epochs: {epochs}, Patience: {patience}")
    
    # Run all experiments
    all_results = []
    
    for aggregator in aggregators:
        # Baseline
        result = run_single_experiment(
            use_structural_features=False,
            aggregator=aggregator,
            device=device,
            epochs=epochs,
            patience=patience
        )
        all_results.append(result)
        
        # Augmented
        result = run_single_experiment(
            use_structural_features=True,
            aggregator=aggregator,
            device=device,
            epochs=epochs,
            patience=patience
        )
        all_results.append(result)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    
    df = pd.DataFrame(all_results)
    print(df.to_string(index=False))
    
    # Calculate improvements
    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS (Augmented vs Baseline)")
    print("=" * 70)
    
    for aggregator in df['aggregator'].unique():
        baseline = df[(df['features'] == 'Baseline') & (df['aggregator'] == aggregator)]
        augmented = df[(df['features'] == 'Augmented') & (df['aggregator'] == aggregator)]
        
        if not baseline.empty and not augmented.empty:
            base_f1 = baseline['test_micro_f1'].values[0]
            aug_f1 = augmented['test_micro_f1'].values[0]
            improvement = aug_f1 - base_f1
            improvement_pct = (improvement / base_f1) * 100
            
            print(f"\n{aggregator}:")
            print(f"  Baseline:  {base_f1:.4f}")
            print(f"  Augmented: {aug_f1:.4f}")
            print(f"  Change:    {improvement:+.4f} ({improvement_pct:+.2f}%)")
    
    print("=" * 70)
    
    # Save results
    df.to_csv('experiment_results.csv', index=False)
    print("\n✓ Results saved to experiment_results.csv")


if __name__ == "__main__":
    main()