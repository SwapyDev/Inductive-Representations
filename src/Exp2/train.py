import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
import numpy as np

class Trainer:
    """
    Trainer for GraphSAGE on PPI dataset.
    """
    def __init__(self, model, device='cpu', lr=0.005):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        # Track history
        self.train_losses = []
        self.val_f1_scores = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(self.device)
            
            # Sort edges for LSTM aggregator (required by PyG)
            data = data.sort(sort_by_row=False)
            
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = self.criterion(out, data.y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def evaluate(self, loader):
        """Evaluate model on a dataset."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        for data in loader:
            data = data.to(self.device)
            
            # Sort edges for LSTM aggregator (required by PyG)
            data = data.sort(sort_by_row=False)
            
            out = self.model(data.x, data.edge_index)
            pred = (torch.sigmoid(out) > 0.5).float()
            
            all_preds.append(pred.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        micro_f1 = f1_score(all_labels, all_preds, average='micro')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        return {'micro_f1': micro_f1, 'macro_f1': macro_f1}
    
    def train(self, train_loader, val_loader, epochs=100, patience=20, verbose=True):
        """Full training loop with early stopping."""
        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None
        
        if verbose:
            print("Starting training...")
            print(f"Epochs: {epochs}, Patience: {patience}")
            print("-" * 60)
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            val_f1 = val_metrics['micro_f1']
            
            # Track
            self.train_losses.append(train_loss)
            self.val_f1_scores.append(val_f1)
            
            # Check if best
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | ★ Best!")
            else:
                patience_counter += 1
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                    print(f"Best validation F1: {best_val_f1:.4f}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if verbose:
                print(f"\nLoaded best model (Val F1: {best_val_f1:.4f})")
        
        return best_val_f1


def train_and_evaluate(model, train_dataset, val_dataset, test_dataset, 
                       device='cpu', epochs=100, patience=20, verbose=True):
    """Complete training and evaluation pipeline."""
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Create trainer
    trainer = Trainer(model, device=device)
    
    # Train
    if verbose:
        print("=" * 60)
        print("TRAINING")
        print("=" * 60)
    
    best_val_f1 = trainer.train(train_loader, val_loader, epochs=epochs, patience=patience, verbose=verbose)
    
    # Test
    if verbose:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)
    
    test_metrics = trainer.evaluate(test_loader)
    
    if verbose:
        print(f"Test Micro-F1: {test_metrics['micro_f1']:.4f}")
        print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
        print("=" * 60)
    
    return {
        'best_val_f1': best_val_f1,
        'test_micro_f1': test_metrics['micro_f1'],
        'test_macro_f1': test_metrics['macro_f1'],
        'train_losses': trainer.train_losses,
        'val_f1_scores': trainer.val_f1_scores
    }