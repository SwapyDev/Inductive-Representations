# reddit_dataloader.py
from torch_geometric.datasets import Reddit
import torch

class RedditDataLoader:
    def __init__(self, root='data/Reddit', useStructuralFeatures=False):
        self.root = root
        self.useStructuralFeatures = useStructuralFeatures
        self._dataset = None
        self._data = None
    def load(self, verbose=False):
        if self._dataset is None:
            self._dataset = Reddit(root=self.root)
            self._data = self._dataset[0]
            
            if self.useStructuralFeatures:
                self._add_structural_features()
            
            if verbose:
                self.print_info()
        
        return self._data
    
    def _add_structural_features(self):
        from torch_geometric.utils import degree
        import networkx as nx
        from torch_geometric.utils import to_networkx
        
        print("Computing structural features (this may take a while)")
        
        # Compute degree
        row, col = self._data.edge_index
        nodeDegree = degree(row, self._data.num_nodes, dtype=torch.float)
        
        # Normalize degree
        maxDegree = nodeDegree.max()
        if maxDegree > 0:
            nodeDegree = nodeDegree / maxDegree
        
        # For Reddit (giant graph), clustering coefficient is slow
        # Stack structural features
        structuralFeatures = nodeDegree.unsqueeze(1)  # Just degree for now
        
        # Concatenate with original features
        self._data.x = torch.cat([self._data.x, structuralFeatures], dim=1)
        
        print(f"Added structural features. New feature dim: {self._data.x.shape[1]}")
    
    def print_info(self):
        print(f"Nodes: {self._data.num_nodes:,}")
        print(f"Edges: {self._data.num_edges:,}")
        print(f"Features: {self._data.num_features}")
        print(f"Classes: {self._dataset.num_classes}")
        print(f"Train nodes: {self._data.train_mask.sum():,}")
        print(f"Val nodes: {self._data.val_mask.sum():,}")
        print(f"Test nodes: {self._data.test_mask.sum():,}")
        print("=" * 60)