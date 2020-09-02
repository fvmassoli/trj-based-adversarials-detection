import sys

import os
import argparse
import itertools
import numpy as np
from tqdm import tqdm, trange
from collections import OrderedDict

from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import *
from adversarials_detector.detector import Detector
from adversarials_detector.embedder import precompute_embeddings


MAIN_OUTPUT_PATH = './adversarial_face_recognition_outputs'


def evaluate(loader, attacks, detector, device):
    detector.eval()

    with torch.no_grad():
        y = []
        y_hat = []
        loss = 0
        correct = 0
        for Xb, yb in tqdm(loader):
            y.append(yb.numpy())
            logits = detector(Xb.to(device))
            yb_hat = F.sigmoid(logits).squeeze().cpu().numpy()

            y_hat.append(yb_hat)

            loss_ = F.binary_cross_entropy_with_logits(logits, yb.reshape(-1, 1).float().to(device))
            loss += loss_.detach().cpu().item()

            ys = F.sigmoid(logits)
            ys[ys > 0.5] = 1.0
            ys[ys <= 0.5] = 0.0
            correct += (ys.detach().cpu().numpy() == yb.reshape(-1, 1).numpy()).astype(np.int32).sum()

        y = np.concatenate(y)
        y_hat = np.concatenate(y_hat)

        # AUC
        fpr, tpr, thr = metrics.roc_curve(y, y_hat)
        auc = metrics.auc(fpr, tpr)

        # EER accuracy
        fnr = 1 - tpr
        eer_thr = thr[np.nanargmin(np.absolute(fnr - fpr))]
        eer_accuracy = metrics.accuracy_score(y, y_hat > eer_thr)
        eer = (eer_accuracy, eer_thr)
        tqdm.write('EER Accuracy: {:3.2%} ({:g})'.format(*eer))

        # Best TPR-FPR
        dist = fpr ** 2 + (1 - tpr) ** 2
        best = np.argmin(dist)
        best = fpr[best], tpr[best], thr[best], auc
        tqdm.write('BEST TPR-FPR: {:4.3%} {:4.3%} ({:g}) AUC: {:4.3%}'.format(*best))

        # Macro-avg AUC
        a = [a_ for a_ in attacks]
        data = pd.DataFrame({'pred': y_hat, 'target': y, 'attack': a})
        auths = data[data.attack == 'natural']
        print('Attack AUCs:')
        aucs = {}
        for attack, group in data.groupby('attack'):
            if attack == 'natural': continue
            pred = np.concatenate((group.pred.values, auths.pred.values))
            target = np.concatenate((group.target.values, auths.target.values))
            aucs[attack] = metrics.roc_auc_score(target, pred)
            print('{}: {:4.3%}'.format(attack, aucs[attack]))

            print(metrics.classification_report(target, np.rint(pred-0.01)))

        macro_auc = sum(aucs.values()) / len(aucs)
        print(f"Macro AUC: {macro_auc:4.3%}")

        return ('auc', 'eer_accuracy', 'macro_auc'), torch.tensor((auc, eer_accuracy, macro_auc)), aucs, loss


def main(args):
    # Set seeds and cuda behaviour for results reproducibility #
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    ## redirecting stdout to local log file
    log_writer = open(os.path.join('./', 'log_eval_'+args.dataset_name+'.txt'), 'a+')
    sys.stdout = sys.stderr = log_writer

    device = 'cuda' 

    model_arc = 'small' if args.dataset_name == 'mnist' else 'wide'
    model_checkpoint = 'models/mnist/model-smallcnn.pth' if  args.dataset_name == 'mnist' else 'models/cifar10/model-wideres.pth'
    model = load_model(model_arc, model_checkpoint)

    ## Register hooks
    def extract(self, input, output):
        features = output
        if output.ndimension() > 2:
            features = F.avg_pool2d(output, output.shape[-2:]).squeeze(3).squeeze(2)
        features_state[self] = features

    if model_arc == 'small':
        blocks = list(itertools.chain((model.feature_extractor.relu2,), (model.feature_extractor.relu3,), (model.feature_extractor.relu4,), (model.classifier.relu1,), (model.classifier.relu2,)))
    else:
        blocks_ = list(itertools.chain(model.block1.layer, model.block2.layer, model.block3.layer, (model.avg_pool,)))
        blocks = [block.relu2 for block in blocks_ if not isinstance(block, nn.AvgPool2d)]
        blocks.append(blocks_[-1])                   
        
    [b.register_forward_hook(extract) for b in blocks]

    best_d = {}
    best_d_bd = {}

    for detector_checkpoint in os.listdir(args.checkpoints_dir):
        detector_checkpoint = os.path.join(MAIN_CKP_PATH, args.dataset_name, 'detector_ckp', detector_checkpoint)

        args.distance = detector_checkpoint.split('/')[-1].split('_')[4].split('-')[-1]
        class_representatives = detector_checkpoint.split('/')[-1].split('_')[3].split('-')[-1]
        detector_arc = detector_checkpoint.split('/')[-1].split('_')[1].split('-')[-1]
        bidir = True if '_bd-' in detector_checkpoint else False
        hidden = 100

        detector = Detector(hidden=hidden, bidir=bidir, architecture=detector_arc, model_arc=model_arc)
        detector.to(device)
        detector.load_state_dict(torch.load(detector_checkpoint)['detector'])

        b = os.path.join(MAIN_OUTPUT_PATH, args.dataset_name, 'class_representatives')
        args.class_representatives_path = os.path.join(b, 'centroids.hdf5') if class_representatives == 'c' else os.path.join(b, 'medoids.hdf5')

        ## Precompute embeddings only once
        root = os.path.join(MAIN_OUTPUT_PATH, 'adversarials', args.dataset_name)
        test_ds = OrigAdvDataset(root=root, split='test')
        test_cache = 'cache_test_{}_{}.pth'.format('cos' if args.distance == 'cosine' else 'euc',
                                                'med' if class_representatives == 'm' else 'centr')
        test_cache_path = os.path.join(MAIN_OUTPUT_PATH, args.dataset_name, 'detector_cache/test', test_cache)
        attacks, test_dataset_embeddings = precompute_embeddings(features_state, test_ds, model, args, test_cache_path, device)

        ## Init loader
        test_loader = DataLoader(test_dataset_embeddings, shuffle=False, pin_memory=True, num_workers=8, batch_size=128)

        ## Eval model
        val_metrics_names, val_metrics, aucs, loss = evaluate(test_loader, attacks, detector, device)
        val_metrics_dict = dict(zip(val_metrics_names, val_metrics.tolist()))
        
        if '_bd-' in detector_checkpoint:
            k = detector_arc+'-'+class_representatives+'-'+args.distance+'-bd'
            if k not in best_d_bd:
                best_d_bd[k] = ["", torch.tensor([0])]
            if best_d_bd[k][1] < val_metrics[2]:
                best_d_bd[k][0] = detector_checkpoint
                best_d_bd[k][1] = val_metrics[2]
        else:
            k = detector_arc+'-'+class_representatives+'-'+args.distance
            if k not in best_d:
                best_d[k] = ["", torch.tensor([0])]
            if best_d[k][1] < val_metrics[2]:
                best_d[k][0] = detector_checkpoint
                best_d[k][1] = val_metrics[2]
            
        del detector
        
    print()
    print()
    print("#"*60)
    print("#"*20, "Best Models", "#"*20)
    print("#"*60)
    for k in best_d.keys():
        print(f"{k}: Macro AUC = {best_d[k]}")
    for k in best_d_bd.keys():
        print(f"{k}: Macro AUC = {best_d_bd[k]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Detection')
    parser.add_argument('-s', '--seed', type=int, default=13, help='Random seed')
    parser.add_argument('-ds', '--dataset-name', choices=('mnist', 'cifar10'), default='mnist')
    args = parser.parse_args()

    args.checkpoints_dir = os.path.join(MAIN_CKP_PATH, args.dataset_name, 'detector_ckp')
    
    global features_state
    features_state = OrderedDict()

    main(args)

