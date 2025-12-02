import torch
from torch_geometric.nn import GraphSAGE
from dataloader import DataLoader as PPIDataLoader
from train import train_and_evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

# Load data
print("\nLoading data...")
loader = PPIDataLoader(useStructuralFeatures=False)
train_dataset, val_dataset, test_dataset = loader.load(verbose=False)

in_channels = train_dataset[0].num_features
print(f"Features: {in_channels}")

# Try lstm with normal configuration, spoiler: it won't work probably
print("\n" + "=" * 50)
print("\n trying with 128 hidden channels...")
print("=" * 50)

try:
    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=128,
        num_layers=2,
        out_channels=121,
        dropout=0.5,
        aggr="lstm",
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nTraining 5 epochs...")
    
    results = train_and_evaluate(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device,
        epochs=5,  # only 5 for quick test
        patience=10,
        verbose=False,
    )
    
    print("\nWORKS!!!!!A KFJKNFL")
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\n Out of memory trying with hiddenchannels=128")
        print("\nhiddenchannels=64...")
        
        #Clean memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        try:
            model = GraphSAGE(
                in_channels=in_channels,
                hidden_channels=64,  # reduced
                num_layers=2,
                out_channels=121,
                dropout=0.5,
                aggr="lstm",
            )
            
            print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print("Training 5 epochs...")
            
            results = train_and_evaluate(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                device=device,
                epochs=5,
                patience=10,
                verbose=False,
            )
            
            print("\nWORKS WITHhidden_channels=64!")
            
        except RuntimeError as e2:
            if "out of memory" in str(e2).lower():
                print("\n Valio madre, probando hidden_channels=64")
            else:
                print(f"\n Error: {e2}")
    else:
        print(f"\n Error: {e}")

print("\n" + "=" * 50)
print("Test finished!")
print("=" * 50)