import sys
sys.path.append('/home/fabiovalerio/lavoro/adversarial_face_recognition/paper-revision')

import os
import argparse
import itertools
import numpy as np
from tqdm import tqdm, trange
from collections import OrderedDict

from sklearn import metrics

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import WeightedRandomSampler

from utils import *
from adversarials_detector.detector import Detector
from adversarials_detector.embedder import precompute_embeddings


MAIN_OUTPUT_PATH = '/media/fabiovalerio/adversarial_face_recognition_revision_outputs'


def train(loader, detector, optimizer, device):
    detector.train()
    progress = tqdm(loader)
    optimizer.zero_grad()
    correct = 0
    for idx, (x, y) in enumerate(progress):

        y = y.reshape(-1, 1).float().to(device)
        y_hat = detector(x.to(device))
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        ys = F.sigmoid(y_hat)
        ys[ys > 0.5] = 1.0
        ys[ys <= 0.5] = 0.0
        correct += (ys.detach().cpu().numpy() == y.detach().cpu().numpy()).astype(np.int32).sum()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress.set_postfix({'loss': '{:6.4f}'.format(loss.tolist())})


def evaluate(loader, attacks, detector, device, test=False):
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
            
            # fpr_, _, thr_ = metrics.roc_curve(target, pred)
            # print(f"attack: {attack}")
            # for fpr_thr in [0.01, 0.03, 0.05, 0.1]:
            #     thr__ = thr_[np.nanargmin(np.absolute(fpr_ - fpr_thr))]
            #     pred_ = pred[target == 1]
            #     pred_[pred_ >= thr__] = 1
            #     pred_[pred_ < thr__] = 0
            #     adv_acc = pred_.sum() / pred[target == 1].shape[0]
            #     print(f"fpr_thr: {fpr_thr} - adv_acc: {adv_acc*100:.2f}%")
            # print()
        
            aucs[attack] = metrics.roc_auc_score(target, pred)
            print('{}: {:4.3%}'.format(attack, aucs[attack]))
            
            if test:
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

    device = 'cuda'
    
    model = load_model(args.model_arc, args.model_checkpoint)
    
    detector = Detector(hidden=args.hidden, bidir=args.bidir, architecture=args.detector_arc, model_arc=args.model_arc)
    detector.to(device)
    if args.test_detector:
        detector.load_state_dict(torch.load(args.detector_checkpoint)['detector'])

    ## Register hooks
    def extract(self, input, output):
        features = output
        if output.ndimension() > 2:
            features = F.avg_pool2d(output, output.shape[-2:]).squeeze(3).squeeze(2)
        features_state[self] = features

    if args.model_arc == 'small':
        blocks = list(itertools.chain((model.feature_extractor.relu2,), (model.feature_extractor.relu3,), (model.feature_extractor.relu4,), (model.classifier.relu1,), (model.classifier.relu2,)))
    else:
        blocks_ = list(itertools.chain(model.block1.layer, model.block2.layer, model.block3.layer, (model.avg_pool,)))
        blocks = [block.relu2 for block in blocks_ if not isinstance(block, nn.AvgPool2d)]
        blocks.append(blocks_[-1])                   
        
    [b.register_forward_hook(extract) for b in blocks]

    root = '/media/fabiovalerio/adversarial_face_recognition_revision_outputs/adversarials'
    
    ## Precompute embeddings only once
    if args.test_detector:
        test_ds = OrigAdvDataset(root=os.path.join(root, args.dataset_name), split='test')

        test_cache = 'cache_test_{}_{}.pth'.format('cos' if args.distance == 'cosine' else 'euc',
                                                'med' if args.class_representatives == 'm' else 'centr')
        test_cache_path = os.path.join(MAIN_OUTPUT_PATH, args.dataset_name, 'detector_cache/test', test_cache)
        attacks, test_dataset_embeddings = precompute_embeddings(features_state, test_ds, model, args, test_cache_path, device)

        test_loader = DataLoader(test_dataset_embeddings, shuffle=False, pin_memory=True, num_workers=8, batch_size=args.batch_size)

    else:   
        train_ds = OrigAdvDataset(root=os.path.join(root, args.dataset_name), split='train')
        val_ds = OrigAdvDataset(root=os.path.join(root, args.dataset_name), split='val')
 
        train_cache = 'cache_train_{}_{}.pth'.format('cos' if args.distance == 'cosine' else 'euc',
                                                    'med' if args.class_representatives == 'm' else 'centr')
        train_cache_path = os.path.join(MAIN_OUTPUT_PATH, args.dataset_name, 'detector_cache/train', train_cache)
        _, train_dataset_embeddings = precompute_embeddings(features_state, train_ds, model, args, train_cache_path, device)
        
        val_cache = 'cache_val_{}_{}.pth'.format('cos' if args.distance == 'cosine' else 'euc',
                                                'med' if args.class_representatives == 'm' else 'centr')
        val_cache_path = os.path.join(MAIN_OUTPUT_PATH, args.dataset_name, 'detector_cache/val', val_cache)
        attacks, val_dataset_embeddings = precompute_embeddings(features_state, val_ds, model, args, val_cache_path, device)
    
        # weights to balance training samples
        weights = train_ds.weights
        sampler = WeightedRandomSampler(weights, len(weights))
        
        ## Init loaders
        train_loader = DataLoader(train_dataset_embeddings, batch_size=args.batch_size, sampler=sampler, pin_memory=True, num_workers=8)
        val_loader   = DataLoader(val_dataset_embeddings,   batch_size=args.batch_size, shuffle=False,   pin_memory=True, num_workers=8)

    optimizer = Adam(detector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1.e-7, threshold=0.1)

    # creates directory to store detector checkpoints
    ckp_dir = os.path.join(MAIN_OUTPUT_PATH, args.dataset_name, 'detector_ckp')
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)

    best = torch.zeros(3)
    progress = trange(1, args.epochs + 1)
    
    if not args.test_detector:
        for epoch in progress:
            progress.set_description("TRAIN")
            train(train_loader, detector, optimizer, device)

            progress.set_description("EVAL")
            val_metrics_names, val_metrics, aucs, loss = evaluate(val_loader, attacks, detector, device)
            scheduler.step(loss, epoch)

            val_metrics_dict = dict(zip(val_metrics_names, val_metrics.tolist()))

            if best[2] < val_metrics[2]:  # keep best macro-AUC
                ckpt_path = os.path.join(ckp_dir, 'detector_arch-'+args.detector_arc+'_dset-'+str(args.dataset_name)+'_cr-'+args.class_representatives+'_dist-'+args.distance+'_lr-'+str(args.learning_rate)+'_wd-'+str(args.weight_decay)+'_bs-'+str(args.batch_size)+'_bd-'+str(args.bidir)+'.pth')
                torch.save({
                    'detector': detector.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'metrics': val_metrics_dict,
                    'aucs': aucs
                }, ckpt_path)

            best = torch.max(val_metrics, best)
        
    else:
        evaluate(test_loader, attacks, detector, device, test=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Detection')
    parser.add_argument('-s', '--seed', type=int, default=13, help='Random seed')
    # DETECTOR PARAMS
    parser.add_argument('-da', '--detector-arc', choices=('mlp', 'lstm'), default='lstm')
    parser.add_argument('-hd', '--hidden', type=int, default=100)
    parser.add_argument('-bd', '--bidir', action='store_true')
    # MODEL PARAMS
    parser.add_argument('-ma', '--model-arc', choices=('small', 'wide'), default='small')
    # TRAIN PARAMS
    parser.add_argument('-tt', '--test-detector', action='store_true', help='Test detector (default: False)')
    parser.add_argument('-dck', '--detector-checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-bs', '--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.0, help='L2 penalty weight decay')
    parser.add_argument('-d', '--distance', choices=('euclidean', 'cosine'), default='euclidean')
    parser.add_argument('-cr', '--class-representatives', choices=('m', 'c'), default='c')
    
    args = parser.parse_args()

    if args.test_detector:
        args.distance = args.detector_checkpoint.split('/')[-1].split('_')[4].split('-')[-1]
        args.class_representatives = args.detector_checkpoint.split('/')[-1].split('_')[3].split('-')[-1]
        args.detector_arc = args.detector_checkpoint.split('/')[-1].split('_')[1].split('-')[-1]
        print("Start test detecor")
        if 'mnist' in args.detector_checkpoint:
            args.model_arc = 'small'
            args.dataset_name = 'mnist'
            args.model_checkpoint = '../toy_models/mnist/model-smallcnn.pth'
        else:
            args.model_arc = 'wide'
            args.dataset_name = 'cifar10'
            args.model_checkpoint = '../toy_models/cifar10/model-wideres.pth'
    else:
        if args.model_arc == 'small':
            args.dataset_name = 'mnist'
            args.model_checkpoint = '../toy_models/mnist/model-smallcnn.pth'
        else:
            args.dataset_name = 'cifar10'
            args.model_checkpoint = '../toy_models/cifar10/model-wideres.pth'

    b = os.path.join('/media/fabiovalerio/adversarial_face_recognition_revision_outputs', args.dataset_name, 'class_representatives')
    args.class_representatives_path = os.path.join(b, 'centroids.hdf5') if args.class_representatives == 'c' else os.path.join(b, 'medoids.hdf5')

    global features_state
    features_state = OrderedDict()

    main(args)

