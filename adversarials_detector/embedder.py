import os
import h5py
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset


class Embedder(nn.Module):
    def __init__(self, class_representatives_path, distance, device):
        super(Embedder, self).__init__()
        self._distance = distance
        with h5py.File(class_representatives_path, 'r') as f:
            self._centroids = [torch.tensor(i[()]).to(device) for i in f.values()]
            
    def _embed(self, x, c):
        if self._distance == 'euclidean':
            return torch.stack([(i - c).norm(dim=1) for i in x])
        else:
            return torch.stack([F.cosine_similarity(i.unsqueeze(0), c) for i in x])

    def forward(self, x):
        embedded = [self._embed(x_i, c_i) for x_i, c_i in zip(x, self._centroids)]
        embedded = torch.stack(embedded)
        return embedded


def precompute_embeddings(features_state, dataset, model, args, cache_path, device):
    if cache_path is not None and os.path.exists(cache_path):
        print(f"Found cache at: {cache_path} ... loading!!!")
        cache = torch.load(cache_path)
        X = cache['X']
        y = cache['y']
        a = cache['a']

    else:
        if not os.path.exists('/'.join(cache_path.split('/')[:-1])):
            os.makedirs('/'.join(cache_path.split('/')[:-1]))
        
        embed = Embedder(args.class_representatives_path, args.distance, device)
        loader = DataLoader(dataset, pin_memory=True, num_workers=8, batch_size=512)
        
        a, X, y = [], [], []
        
        with torch.no_grad():
            for imgs, labels, attacks in tqdm(loader, total=len(loader), desc='Crating cache', leave=False):
                a.extend(attacks)

                y.append(labels.cpu())
                model(imgs.to(device))
                
                features = list(features_state.values())
                embedded = embed(features).permute(1, 0, 2)
                X.append(embedded.cpu())

        del embed, loader
        X = torch.cat(X)
        y = torch.cat(y)
        
        torch.save({'X': X, 'y': y, 'a': a}, cache_path)

    return a, TensorDataset(X, y)

