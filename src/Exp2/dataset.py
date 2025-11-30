from torch_geometric.datasets import PPI
from torch_geometric.utils import degree
from torch_geometric.transforms import BaseTransform
import torch 
from torch_geometric.utils import to_networkx
import networkx as nx 

#Use the PPI dataset from PyTorch Geometric
class DataLoader:
    def __init__(self, root = 'data/PPI', useStructuralFeatures = False):
        self.root = root
        self.useStructuralFeatures = useStructuralFeatures
        self._train = None
        self._val = None
        self._test = None

        self.transform = AddStructuralFeatures(normalize=True) if useStructuralFeatures else None

    def load(self, verbose = False):
        #Load all the splits
        if self._train is None:
            self._train = PPI(root=self.root, split='train', transform=self.transform)
            self._val  = PPI(root=self.root, split='val', transform=self.transform)
            self._test  = PPI(root=self.root, split='test', transform=self.transform)
        
            if verbose:
                self.print_info()

        return self._train, self._val, self._test
    
    @property
    def train(self):
        if self._train is None:
            self.load()
        return self._train 
    
    @property
    def val(self):
        if self._val is None:
            self.load()
        return self._val
    
    @property
    def test(self):
        if self._test is None:
            self.load()
        return self._test
    #Dataset info 
    def print_info(self):
        print("PPI Dataset Info")
        print(f"Train graphs: {len(self.train)}")
        print(f"Val graphs:   {len(self.val)}")
        print(f"Test graphs:  {len(self.test)}")

        sample = self._train[0]
        print("\nSample Graph:")
        print(f"  Nodes: {sample.num_nodes}")
        print(f"  Edges: {sample.num_edges}")
        print(f"  Features: {sample.num_features}")
        print(f"  Labels: {sample.y.size(1)}")
        
        deg = degree(sample.edge_index[0], sample.num_nodes)
        print("\nDegree Stats:")
        print(f"  Average: {deg.mean():.2f}")
        print(f"  Max: {deg.max().item()}")
        print(f"  Min: {deg.min().item()}")

class AddStructuralFeatures(BaseTransform):
    def __init__(self, normalize = True):
        self.normalize = normalize

    #__call__ makes an object callable (i didn't know that before, lol) nvm, use forward instead of __call__
    def forward(self, data):
        row, col = data.edge_index
        #This counts how many times each node appears in the edgeIndex
        nodeDegree = degree(row, data.num_nodes, dtype=torch.float)

        #compute clustering coefficient
        clusteringCoef = self.ComputeClusteringCoefficient(data)

        #Normalize features if needed
        if self.normalize:
            maxNodeDegree = nodeDegree.max()
            #Normalize degree to [0,1]
            if maxNodeDegree > 0:
                nodeDegree = nodeDegree / maxNodeDegree

        #Stack features together
        structuralFeatures = torch.stack([nodeDegree, clusteringCoef], dim=1)

        #Append structural features to the existing node features
        data.x = torch.cat([data.x, structuralFeatures], dim=1)

        return data

    def ComputeClusteringCoefficient(self, data):
        G = to_networkx(data, to_undirected = True)

        #compute clustering coefficient
        clusteringDict = nx.clustering(G)

        clusteringCoef = torch.tensor(
            [clusteringDict[i] for i in range(data.num_nodes)],
            dtype=torch.float
        )

        return clusteringCoef
    
#test it out 

if __name__ == "__main__":
    print("Baseline (No Structural Features)")
    loader_baseline = DataLoader(useStructuralFeatures=False)
    train_base, val_base, test_base = loader_baseline.load(verbose=True)
    
    print ("-" * 40)
    print("Augmented (With Structural Features)")
    loader_augmented = DataLoader(useStructuralFeatures=True)
    train_aug, val_aug, test_aug = loader_augmented.load(verbose=True)
    
    print ("-" * 40)
    print("Comparison of feature dimensions:")
    print(f"Baseline feature dimension: {train_base[0].num_features}")
    print(f"Augmented feature dimension: {train_aug[0].num_features}")
    print(f"Difference: +{train_aug[0].num_features - train_base[0].num_features} features")