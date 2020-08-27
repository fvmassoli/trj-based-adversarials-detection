import os
import h5py
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset


class Embedder(nn.Module):
    def __init__(self, centroids_path, distance, device):
        super(Embedder, self).__init__()
        self._distance = distance
        with h5py.File(centroids_path, 'r') as f:
            self._centroids = [torch.tensor(i[()]).to(device) for i in f.values() if 'classifier' not in i.name and 'avg_pool' not in i.name]
            
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
            self.fc = nn.Sequential(nn.Linear(in_features=8000, out_features=hidden),
                                    nn.ReLU(),
                                    nn.Dropout(0.5)
                                )
        else:
            print('Initializing LSTM')
            self.lstm = nn.LSTM(input_size=500, hidden_size=hidden, bidirectional=bidir, batch_first=True)

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


class ThreatenedModelVGGFace(nn.Module):
    """ Classifier + Embedder + Detector """
    def __init__(self, centroids_path, distance, device, detector_ckp, hidden, bidir, architecture, model_ckp):
        super(ThreatenedModelVGGFace, self).__init__()
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
        model = torch.load('./senet50_ft_pytorch.pth', map_location=lambda storage, loc: storage)
        model.classifier_1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features=2048, out_features=500, bias=True))
        ckp = torch.load(model_ckp, map_location='cpu')
        [p.data.copy_(torch.from_numpy(ckp['model_state_dict'][n].numpy())) for n, p in model.named_parameters()]
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.1
                m.running_var = ckp['model_state_dict'][n + '.running_var']
                m.running_mean = ckp['model_state_dict'][n + '.running_mean']
                m.num_batches_tracked = ckp['model_state_dict'][n + '.num_batches_tracked']
        return model.eval().to(self._device)

    def forward(self, x, return_trajectory=False):
        ## forward on classifier to collect logits and features
        classifier_outputs = self._classifier(x)
        feat = []
        for feat_ in classifier_outputs[:-1]:
            if feat_.ndimension() > 2:
                feat.append(F.avg_pool2d(feat_, feat_.shape[-2:]).squeeze(3).squeeze(2))
            else:
                feat.append(feat_)
        ## forward on embedder to obtain trajectories
        embedded = self._embedder(feat).permute(1, 0, 2)
        ## forward on detector 
        detector_out = self._detector(embedded)
        ## return classifier logits and detector output
        if not return_trajectory:
            return classifier_outputs[-1], detector_out
        else:
            return feat

    def extract_features(self, x):
        classifier_outputs = self._classifier(x)
        features = classifier_outputs[-2].squeeze()
        assert features.shape[-1] == 2048
        return features






