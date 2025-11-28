from torch_geometric.datasets import PPI
from torch_geometric.utils import degree
#Use the PPI dataset from PyTorch Geometric
class DataLoader:
    def __init__(self, root = 'data/PPI'):
        self.root = root
        self._train = None
        self._val = None
        self._test = None

    def load(self, verbose = False):
        #Load all the splits
        if self._train is None:
            self._train = PPI(root=self.root, split='train')
            self._val  = PPI(root=self.root, split='val')
            self._test  = PPI(root=self.root, split='test')
        
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
    
ppiDataLoader = DataLoader()
ppiDataLoader.load(verbose=True)