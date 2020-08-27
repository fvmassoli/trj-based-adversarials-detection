import sys
sys.path.append('../')

import os
import h5py
import itertools
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from toy_models.mnist.small_cnn import SmallCNN


features_state = OrderedDict()


class Embedder(nn.Module):
    def __init__(self, centroids_path, distance, device):
        super(Embedder, self).__init__()
        self._distance = distance
        with h5py.File(centroids_path, 'r') as f:
            self._centroids = [torch.tensor(i[()]).to(device) for i in f.values()]
            
    def _embed(self, x, c):
        if self._distance == 'euclidean':
            return torch.stack([(i - c).norm(dim=1) for i in x])
        elif self._distance == 'cosine':
            return torch.stack([F.cosine_similarity(i.unsqueeze(0), c) for i in x])

    def forward(self, x):
        embedded = [self._embed(x_i, c_i) for x_i, c_i in zip(x, self._centroids)]
        embedded = torch.stack(embedded)
        return embedded


class Detector(nn.Module):
    def __init__(self, hidden=100, bidir=False, architecture=None):
        super(Detector, self).__init__()
        self.mlp = True if architecture == 'mlp' else False

        if architecture == 'mlp':
            print('Initializing MLP')
            self.fc = nn.Sequential(nn.Linear(in_features=50, out_features=hidden),
                                    nn.ReLU(),
                                    nn.Dropout(0.5)
                                )
        else:
            print('Initializing LSTM')
            self.lstm = nn.LSTM(input_size=10, hidden_size=hidden, bidirectional=bidir, batch_first=True)

        out_size = hidden * 2 if (bidir and architecture=='lstm') else hidden
        self.classifier = nn.Linear(out_size, 1)

    def forward(self, x):
        if self.mlp:
            x = x.reshape(x.shape[0], -1)
            output = self.fc(x)
        else:
            output, _ = self.lstm(x)
            output = output[:, -1]
        return self.classifier(output)


class ThreatenedModelMNIST(nn.Module):
    """ Classifier + Embedder + Detector """
    def __init__(self, centroids_path, distance, device, detector_ckp, hidden, bidir, architecture, model_ckp):
        super(ThreatenedModelMNIST, self).__init__()
        self._device = device
        self._feat_d = {}
        ## Init Embedder
        self._embedder = Embedder(centroids_path, distance, device)
        ## Init Detector
        self._detector = self._load_detector(detector_ckp, hidden, bidir, architecture)
        ## Init Classifier
        self._classifier = self._load_classifier(model_ckp)
        
    def _load_detector(self, detector_ckp, hidden, bidir, architecture):
        detector = Detector(hidden, bidir, architecture)
        ckpt = torch.load(detector_ckp, map_location=lambda storage, loc: storage)
        detector.load_state_dict(ckpt['detector'])
        return detector.to(self._device)

    def _load_classifier(self, model_ckp):
        model = SmallCNN()
        model.load_state_dict(torch.load(model_ckp)['model_state_dict'])
        ## Extraction method
        def extract(self, input, output):
            features = output
            if output.ndimension() > 2:
                features = F.avg_pool2d(output, output.shape[-2:]).squeeze(3).squeeze(2)
            features_state[self] = features
        ## Register hooks
        blocks = list(itertools.chain((model.feature_extractor.relu2,), (model.feature_extractor.relu3,), (model.feature_extractor.relu4,), (model.classifier.relu1,), (model.classifier.relu2,)))        
        [b.register_forward_hook(extract) for b in blocks]
        ##
        return model.eval().to(self._device)

    def forward(self, x):
        ## forward on classifier to collect logits and features
        classifier_outputs = self._classifier(x)
        ## forward on embedder to obtain trajectories
        embedded = self._embedder(list(features_state.values())).permute(1, 0, 2)
        ## forward on detector 
        detector_out = self._detector(embedded)
        ## return classifier logits and detector output
        return classifier_outputs, detector_out

    def extract_features(self, x):
        self._classifier(x)
        return list(features_state.values())[-1]







