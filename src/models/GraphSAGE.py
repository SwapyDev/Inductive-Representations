from torch_geometric.nn import GraphSAGE

model = GraphSAGE(
    in_channels=50,
    hidden_channels=256,
    num_layers=3,
    out_channels=121,
    dropout=0.5,
    aggr='mean'  #Aggregation method (mean, max, etc)
)




