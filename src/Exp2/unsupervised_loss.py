# unsupervised_loss.py
import torch
import torch.nn.functional as F
import random

class UnsupervisedLoss:
    def __init__(self, num_neg_samples=10):

        self.num_neg_samples = num_neg_samples
    
    def __call__(self, embeddings, pos_pairs, neg_nodes):
        loss = 0.0
        
        for u, v in pos_pairs:
            z_u = embeddings[u]
            z_v = embeddings[v]
            
            # Positive loss: nearby nodes should have similar embeddings
            pos_score = torch.sigmoid((z_u * z_v).sum())
            pos_loss = -torch.log(pos_score + 1e-8)
            
            # Negative loss: distant nodes should have dissimilar embeddings
            neg_loss = 0.0
            for neg_v in neg_nodes[u]:
                z_neg = embeddings[neg_v]
                neg_score = torch.sigmoid(-(z_u * z_neg).sum())
                neg_loss += -torch.log(neg_score + 1e-8)
            
            loss += pos_loss + neg_loss
        
        return loss / len(pos_pairs)


def generate_random_walks(edge_index, num_nodes, walk_length=5, num_walks=50):
    # Build adjacency list
    adj = {i: [] for i in range(num_nodes)}
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        adj[u].append(v)
        adj[v].append(u)  # Undirected
    
    walks = []
    for node in range(num_nodes):
        for _ in range(num_walks):
            walk = [node]
            current = node
            
            for _ in range(walk_length - 1):
                neighbors = adj[current]
                if len(neighbors) == 0:
                    break
                current = random.choice(neighbors)
                walk.append(current)
            
            walks.append(walk)
    
    return walks


def walks_to_pairs(walks, window_size=5):
    pairs = []
    
    for walk in walks:
        for i, u in enumerate(walk):
            # Get context nodes within window
            start = max(0, i - window_size)
            end = min(len(walk), i + window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    v = walk[j]
                    pairs.append((u, v))
    
    return pairs


def sample_negative_nodes(num_nodes, num_neg_samples, node_degrees=None):
    # Negative sampling with degree-based distribution (as in paper)
    if node_degrees is not None:
        # Sample based on degree^0.75 (same as word2vec)
        probs = torch.pow(node_degrees.float(), 0.75)
        probs = probs / probs.sum()
    else:
        probs = None
    
    neg_samples = {}
    for u in range(num_nodes):
        if probs is not None:
            neg_nodes = torch.multinomial(probs, num_neg_samples, replacement=True).tolist()
        else:
            neg_nodes = random.sample(range(num_nodes), num_neg_samples)
        neg_samples[u] = neg_nodes
    
    return neg_samples