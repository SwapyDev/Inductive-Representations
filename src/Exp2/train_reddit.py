import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

class SimpleRedditTrainer:
    """Simple trainer for Reddit without NeighborLoader"""
    
    def __init__(self, model, data, device='cpu', lr=0.01):
        self.model = model.to(device)
        self.device = device
        self.data = data.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(self.data.x, self.data.edge_index)
        loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, mask):
        """Evaluate on nodes specified by mask"""
        self.model.eval()
        
        out = self.model(self.data.x, self.data.edge_index)
        pred = out[mask].argmax(dim=1)
        y_true = self.data.y[mask]
        
        f1_micro = f1_score(y_true.cpu(), pred.cpu(), average='micro')
        f1_macro = f1_score(y_true.cpu(), pred.cpu(), average='macro')
        accuracy = (pred == y_true).sum().item() / mask.sum().item()
        
        return {
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro
        }
    
    def train(self, epochs=100, patience=20, verbose=True):
        """Full training loop"""
        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None
        
        if verbose:
            print("Starting training...")
            print(f"Epochs: {epochs}, Patience: {patience}")
            print("-" * 60)
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.data.val_mask)
            val_f1 = val_metrics['f1_micro']
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                
                if verbose and epoch % 5 == 0:
                    print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | ★ Best!")
            else:
                patience_counter += 1
                if verbose and epoch % 5 == 0:
                    print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                    print(f"Best validation F1: {best_val_f1:.4f}")
                break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        if verbose:
            print("\n" + "=" * 60)
            print("FINAL EVALUATION")
            print("=" * 60)
        
        test_metrics = self.evaluate(self.data.test_mask)
        
        if verbose:
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Test F1 (micro): {test_metrics['f1_micro']:.4f}")
            print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
            print("=" * 60)
        
        return test_metrics