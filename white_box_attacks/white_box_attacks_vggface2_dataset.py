import os
import sys
import cv2
from PIL import Image

import torchvision.transforms as t
from torch.utils.data import Dataset


def subtract_mean(x):
    mean_vector = [91.4953, 103.8827, 131.0912]
    x *= 255.
    x[0] -= mean_vector[0]
    x[1] -= mean_vector[1]
    x[2] -= mean_vector[2]
    return x


class CustomDataset(Dataset):
    def __init__(self, root, imgs_dict):
        self._root = root
        self._imgs = [img for imgs in imgs_dict.values() for img in imgs]
        self._transform = self._transforms()
        self._loader = self._get_loader
        self._classes, self._class_to_idx = self._find_classes()
        
    @staticmethod
    def _get_loader(path):
        return Image.fromarray(cv2.imread(path))

    @staticmethod
    def _transforms():
        tf = t.Compose([
            t.Resize(256),
            t.CenterCrop(224),
            t.ToTensor(),
            t.Lambda(lambda x: subtract_mean(x))
        ])
        return tf

    def _find_classes(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self._root) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self._root) if os.path.isdir(os.path.join(self._root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, item):
        x = self._transform(self._loader(self._imgs[item]))
        y = self._class_to_idx[self._imgs[item].split('/')[-2]]
        return x, y

    def __len__(self):
        return len(self._imgs)