import sys
sys.path.append('../')

import os
import numpy as np
import pandas as pd
from random import getrandbits
from collections import Counter

from sklearn import metrics
from scipy.spatial.distance import pdist, squareform

import tensorflow as tf

import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, Dataset

from toy_models.mnist.small_cnn import SmallCNN
from toy_models.cifar10.wide import WideResNet


def load_model(model_arc, model_checkpoint):
    model = SmallCNN() if model_arc == 'small' else WideResNet()
    state_dict = torch.load(model_checkpoint, map_location=lambda storage, loc: storage)['model_state_dict']
    model.load_state_dict(state_dict)
    return model.eval().cuda()


def get_loader_for_features_extraction(root, dataset_name):
    if dataset_name == 'mnist':
        dataset = MNIST(root=os.path.join(root, dataset_name), train=True, transform=ToTensor())
    else:
        dataset = CIFAR10(root=os.path.join(root, dataset_name), train=True, transform=ToTensor())
    return DataLoader(dataset=dataset, batch_size=128, num_workers=8, pin_memory=True) 


def get_loader_for_adversarials_generation(root, dataset_name):
    if dataset_name == 'mnist':
        dataset = MNIST(root=os.path.join(root, dataset_name), train=False, transform=ToTensor())
    else:
        dataset = CIFAR10(root=os.path.join(root, dataset_name), train=False, transform=ToTensor())
    return DataLoader(dataset=dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True) 


def compute_centroid(class_representatives, features):
    if class_representatives == 'm':
        distances = pdist(features, metric='euclidean')
        distances = squareform(distances)
        c_idx = distances.sum(axis=1).argmin()
        c = features[c_idx]
    else:
        c = np.mean(features, axis=0)
        c_idx = -100
    return c, c_idx


def generate_targets(model, x, y, targeted):
    # Generate random target for targeted attacks
    nb_classes = 10
    if targeted:
        labels = np.random.randint(0, nb_classes, y.shape[0])
        indices = np.where(labels == y)[0]
        if len(indices) != 0:
            for i in indices:
                labels[i] = (y[i]+1)%nb_classes
    else:
        # Use model predictions as correct outputs
        preds = model(x.cuda())
        preds_max = np.amax(preds.detach().cpu().numpy(), axis=1, keepdims=True)
        targets = preds.detach().cpu().numpy() == preds_max
        labels = targets.argmax(axis=-1)
    return labels


def get_data_splits(root, split=None):
    """Creates csv for train, val, and test splits """
    o_train = os.path.join(root, 'train_split.csv')
    o_val = os.path.join(root, 'val_split.csv')
    o_test = os.path.join(root, 'test_split.csv')
    split_d = dict(train=o_train, val=o_val, test=o_test)
    if os.path.exists(o_train): 
        ## We already generated them
        if split is None:
            return pd.read_csv(o_train), pd.read_csv(o_val), pd.read_csv(o_test)
        else:
            return pd.read_csv(split_d[split])
    ## Concat all frames and shuffle them
    frames = [pd.read_csv(os.path.join(root, f)).sample(frac=1).reset_index(drop=True) for f in os.listdir(root) if '.csv' in f]
    res_ = pd.concat(frames)
    res_ = res_.sample(frac=1).reset_index(drop=True)
    res_ = res_.sample(frac=1).reset_index(drop=True)

    train_df = pd.DataFrame(columns=res_.columns)
    val_df = pd.DataFrame(columns=res_.columns)
    test_df = pd.DataFrame(columns=res_.columns)
    for atk in res_['attack_name'].unique():
        if atk != 'natural':
            train_df = train_df.append(res_[res_['attack_name']==atk][:10000])
            val_df = val_df.append(res_[res_['attack_name']==atk][10000:10100])
            test_df = test_df.append(res_[res_['attack_name']==atk][-1000:])
        else:
            train_df = train_df.append(res_[res_['attack_name']==atk][:8500])
            val_df = val_df.append(res_[res_['attack_name']==atk][8500:8600])
            test_df = test_df.append(res_[res_['attack_name']==atk][8600:])
    train_df.to_csv(o_train)
    val_df.to_csv(o_val)
    test_df.to_csv(o_test)
    
    if split is None:
        return train_df, val_df, test_df
    else:
        return pd.read_csv(split_d[split])


class OrigAdvDataset(Dataset):
    def __init__(self, root, split):
        self.res = get_data_splits(root, split)
        self.res = self.res.sample(frac=1).reset_index(drop=True)
        self.res = self.res.sample(frac=1).reset_index(drop=True)
        self._paths = self.res['path'].tolist()
        self._attacks = self.res['attack_name'].tolist()
        self._labels = [0 if 'nat-' in row['path'] else 1 for idx, row in self.res.iterrows()]
        
        a_counts = Counter(self._attacks)
        ntotal = len(self._paths)
        self.weights = [(ntotal / a_counts[a]) for a in self._attacks]
        
    def __len__(self):
        return len(self._paths)
        
    def __getitem__(self, item):
        return torch.from_numpy(np.transpose(np.load(self._paths[item]+'.npz')['img'], (2, 0, 1))), self._labels[item], self._attacks[item]