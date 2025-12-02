import torch
from torch_geometric.datasets import PPI
from sklearn.metrics import f1_score

# Load PPI
train = PPI(root="data/PPI", split="train")
val = PPI(root="data/PPI", split="val")
test = PPI(root="data/PPI", split="test")


def stack(dataset):
    xs, ys = [], []
    for data in dataset:
        xs.append(data.x)
        ys.append(data.y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


X_train, Y_train = stack(train)
X_val, Y_val = stack(val)
X_test, Y_test = stack(test)

in_dim = X_train.size(1)
out_dim = Y_train.size(1)

# 2. MLP no graph
model = torch.nn.Sequential(
    torch.nn.Linear(in_dim, 256), torch.nn.ReLU(), torch.nn.Linear(256, out_dim)
)

criterion = torch.nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.005)

#  20 epochs
for epoch in range(1, 21):
    model.train()
    opt.zero_grad()
    out = model(X_train)
    loss = criterion(out, Y_train)
    loss.backward()
    opt.step()

# evauate test
model.eval()
with torch.no_grad():
    logits = model(X_test)
    preds = (torch.sigmoid(logits) > 0.5).float()

micro_f1 = f1_score(Y_test.numpy(), preds.numpy(), average="micro")
macro_f1 = f1_score(Y_test.numpy(), preds.numpy(), average="macro")

print(f"Raw features baseline - Test micro-F1: {micro_f1:.4f}")
print(f"Raw features baseline - Test macro-F1: {macro_f1:.4f}")
