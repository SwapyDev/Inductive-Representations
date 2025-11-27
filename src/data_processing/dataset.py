from torch_geometric.datasets import PPI
from torch_geometric.utils import degree
#Use the PPI dataset from PyTorch Geometric
class DataLoader:
    def __init__(self, root = 'data/PPI'):
        self.root = root
        self.train = None
        self.val = None
        self.test = None

    def load(self, verbose = False):
        #Load all the splits
        if self.train is None:
            self.train = PPI(root=self.root, split='train')
            self.val  = PPI(root=self.root, split='val')
            self.test  = PPI(root=self.root, split='test')
        
            if verbose:
                self.print_info()

        return self.train, self.val, self.test
    
    @property
    def train(self):
        if self.train is None:
            self.load()
        return self.train 
    
    @property
    def val(self):
        if self.val is None:
            self.load()
        return self.val
    
    @property
    def test(self):
        if self.tests is None:
            self.load()
        return self.test

    def print_info(self):
        print("PPI Dataset Info")
        print(f"Train graphs: {len(self.train)}")
        print(f"Val graphs:   {len(self.val)}")
        print(f"Test graphs:  {len(self.test)}")
    
        

DataLoader = DataLoader()
