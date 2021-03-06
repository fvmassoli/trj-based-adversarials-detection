import os
import sys
import cv2
import pickle
import pandas as pd
from tqdm import tqdm
from PIL import Image
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms as t
from torchvision.datasets.folder import IMG_EXTENSIONS


class TinyDataset(Dataset):
    def __init__(self, root, transform=None):
        self._root = root
        self._transform = transform
        self._loader = self._get_loader
        self._imgs = os.listdir(root)

    @staticmethod
    def _get_loader(path):
        return Image.fromarray(cv2.imread(path))

    def __getitem__(self, item):
        path = os.path.join(self._root, self._imgs[item])
        img = self._loader(path)
        if self._transform is not None:
            img = self._transform(img)
        return img

    def __len__(self):
        return len(self._imgs)


class CustomDataset(Dataset):
    def __init__(self, root, transform=None, return_paths=False):
        self._root = root
        self._transform = transform
        self._return_paths = return_paths
        self._loader = self._get_loader
        self._classes, self._class_to_idx = self._find_classes()
        self._samples = self._make_dataset()

    @staticmethod
    def _get_loader(path):
        return Image.fromarray(cv2.imread(path))

    def _make_dataset(self):
        images = []
        dir = os.path.expanduser(self._root)
        progress_bar = tqdm(sorted(self._class_to_idx.keys()),
                            desc='Making for features extraction',
                            total=len(self._class_to_idx.keys()),
                            leave=False)
        for target in progress_bar:
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self._class_to_idx[target])
                    images.append(item)
            progress_bar.update(n=1)
        progress_bar.close()
        return images

    def _find_classes(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self._root) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self._root) if os.path.isdir(os.path.join(self._root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self._samples[index]
        image = self._loader(path)
        if self._transform is not None:
            image = self._transform(image)
        if self._return_paths:
            return image, target, path
        else:
            return image, target

    def __len__(self):
        return len(self._samples)


class CustomDatasetFromCsv(Dataset):
    def __init__(self, root, csv_root, transform=None, return_path=False):
        self._root = root
        self._csv_root = csv_root
        self._transform = transform
        self._return_path = return_path
        self._classes, self._class_to_idx = self._find_classes()
        self._cw2_imgs_paths = self._get_imgs_paths('cw2_imgs_path.csv')
        self._bim_imgs_paths = self._get_imgs_paths('bim_imgs_path.csv')
        self._mi_fgsm_imgs_paths = self._get_imgs_paths('mi_fgsm_imgs_path.csv')
        self._imgs_list = self._get_full_imgs_list()
        self._loader = self._get_loader

    def _get_imgs_paths(self, csv_path):
        df = pd.read_csv(os.path.join(self._csv_root, csv_path))
        return df['path'].tolist()

    def _get_full_imgs_list(self):
        imgs_list = self._cw2_imgs_paths
        imgs_list.extend(self._bim_imgs_paths)
        imgs_list.extend(self._mi_fgsm_imgs_paths)
        imgs_list.sort(key=lambda x: x.split('/')[-2])
        return imgs_list

    @staticmethod
    def _get_loader(path):
        return Image.fromarray(cv2.imread(path))

    def _find_classes(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self._root) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self._root) if os.path.isdir(os.path.join(self._root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path = self._imgs_list[index]
        target = self._class_to_idx[path.split('/')[-2]]
        image = self._loader(path)
        if self._transform is not None:
            image = self._transform(image)
        if self._return_path:
            return image, target, path
        else:
            return image, target

    def __len__(self):
        return len(self._imgs_list)


def load_model(base_model_path, model_checkpoint, device):
    model = torch.load(path, map_location=lambda storage, loc: storage, pickle_module=pickle)
    linear = nn.Linear(in_features=2048, out_features=500, bias=True)
    model.classifier_1 = linear
    if model_checkpoint is not None:
        ckp = torch.load(ckp_path, map_location='cpu')
        print('\nLoaded checkpoint from: {} --- with best accuracy of: {:.2f}'.format(ckp_path, ckp['best_acc']))
        [p.data.copy_(torch.from_numpy(ckp['model_state_dict'][n].numpy())) for n, p in model.named_parameters()]
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.1
                m.running_var = ckp['model_state_dict'][n + '.running_var']
                m.running_mean = ckp['model_state_dict'][n + '.running_mean']
                m.num_batches_tracked = ckp['model_state_dict'][n + '.num_batches_tracked']
    return model.eval().to(device)


def subtract_mean(x):
    mean_vector = [91.4953, 103.8827, 131.0912]
    x *= 255.
    x[0] -= mean_vector[0]
    x[1] -= mean_vector[1]
    x[2] -= mean_vector[2]
    return x


def get_transforms():
    tf = t.Compose([
        t.Resize(256),
        t.CenterCrop(224),
        t.ToTensor(),
        t.Lambda(lambda x: subtract_mean(x))
    ])
    return tf


def get_adv_transforms():
    tf = t.Compose([
        t.Resize(256),
        t.CenterCrop(224),
        t.ToTensor(),
        t.Lambda(lambda x: x.numpy()*255.)
    ])
    return tf


def eval_centroids_medoids(features, compute_class_representatives):
    if compute_class_representatives == 'm':
        distances = pdist(features, metric='euclidean')
        distances = squareform(distances)
        c_idx = distances.sum(axis=1).argmin()
        c = features[c_idx]
    else:
        c = np.mean(features, axis=0)
    return c
