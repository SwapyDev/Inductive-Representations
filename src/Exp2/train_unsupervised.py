# train_unsupervised.py
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
import numpy as np
from unsupervised_loss import (
    UnsupervisedLoss, 
    generate_random_walks, 
    walks_to_pairs,
    sample_negative_nodes
)

class UnsupervisedTrainer:
    """
    Unsupervised trainer for GraphSAGE.
    
    1. Train GraphSAGE with unsupervised loss (random walks)
    2. Use learned embeddings to train a separate classifier
    """
    
    def __init__(self, model, device='cpu', lr=0.005):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.unsup_loss = UnsupervisedLoss(num_neg_samples=10)
        
    def train_epoch(self, train_loader):
        """Train for one epoch with unsupervised loss"""
        self.model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(self.device)
            
            # Generate embeddings
            embeddings = self.model(data.x, data.edge_index)
            
            # Generate random walks and pairs
            walks = generate_random_walks(
                data.edge_index.cpu(), 
                data.num_nodes,
                walk_length=5,
                num_walks=10  # Reduced for speed
            )
            pairs = walks_to_pairs(walks, window_size=5)
            
            # Sample negative nodes
            from torch_geometric.utils import degree
            node_degrees = degree(data.edge_index[0], data.num_nodes)
            neg_samples = sample_negative_nodes(
                data.num_nodes, 
                num_neg_samples=10,
                node_degrees=node_degrees
            )
            
            # Compute unsupervised loss
            self.optimizer.zero_grad()
            loss = self.unsup_loss(embeddings, pairs, neg_samples)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def generate_embeddings(self, loader):
        """Generate embeddings for all nodes"""
        self.model.eval()
        
        all_embeddings = []
        all_labels = []
        
        for data in loader:
            data = data.to(self.device)
            embeddings = self.model(data.x, data.edge_index)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        return embeddings, labels
    
    def train_classifier(self, train_embeddings, train_labels):
        """Train a logistic regression classifier on embeddings"""
        # Multi-label classification: one classifier per label
        classifiers = []
        
        for i in range(train_labels.shape[1]):
            clf = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)
            clf.fit(train_embeddings, train_labels[:, i])
            classifiers.append(clf)
        
        return classifiers
    
    def evaluate_with_classifier(self, classifiers, embeddings, labels):
        """Evaluate using the trained classifiers"""
        predictions = []
        
        for clf in classifiers:
            pred = clf.predict(embeddings)
            predictions.append(pred)
        
        predictions = np.stack(predictions, axis=1)
        
        # Compute F1 scores
        micro_f1 = f1_score(labels, predictions, average='micro')
        macro_f1 = f1_score(labels, predictions, average='macro')
        
        return {
            'micro_f1': micro_f1,
            'macro_f1': macro_f1
        }
    
    def train(self, train_loader, val_loader, epochs=100, verbose=True):
        if verbose:
            print("Phase 1: Unsupervised embedding training...")
            print(f"Epochs: {epochs}")
            print("-" * 60)
        
        # Phase 1: Train embeddings with unsupervised loss
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Unsupervised Loss: {train_loss:.4f}")
        
        if verbose:
            print("\nPhase 2: Training classifier on embeddings...")
        
        # Phase 2: Generate embeddings
        train_embeddings, train_labels = self.generate_embeddings(train_loader)
        val_embeddings, val_labels = self.generate_embeddings(val_loader)
        
        # Phase 3: Train classifier
        classifiers = self.train_classifier(train_embeddings, train_labels)
        
        # Evaluate on validation
        val_metrics = self.evaluate_with_classifier(classifiers, val_embeddings, val_labels)
        
        if verbose:
            print(f"Validation F1 (micro): {val_metrics['micro_f1']:.4f}")
            print(f"Validation F1 (macro): {val_metrics['macro_f1']:.4f}")
        
        return classifiers


def train_and_evaluate_unsupervised(
    model, train_dataset, val_dataset, test_dataset,
    device='cpu', epochs=100, verbose=True
):
    #Complete unsupervised training and evaluation
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Train
    trainer = UnsupervisedTrainer(model, device=device)
    
    if verbose:
        print("=" * 60)
        print("UNSUPERVISED TRAINING")
        print("=" * 60)
    
    classifiers = trainer.train(train_loader, val_loader, epochs=epochs, verbose=verbose)
    
    # Test
    if verbose:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)
    
    test_embeddings, test_labels = trainer.generate_embeddings(test_loader)
    test_metrics = trainer.evaluate_with_classifier(classifiers, test_embeddings, test_labels)
    
    if verbose:
        print(f"Test Micro-F1: {test_metrics['micro_f1']:.4f}")
        print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
        print("=" * 60)
    
    return test_metrics