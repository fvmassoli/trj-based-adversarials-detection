import os
import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from adv_attacks.attacks import attack
from white_box_attacks_dataset import CustomDataset
from threatened_model_mnist import ThreatenedModelMNIST
from threatened_model_vggface2 import ThreatenedModelVGGFace

from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


MAIN_OUTPUT = './'


def get_threatened_model(args, device):
    model_cls = ThreatenedModelVGGFace if args.dataset_name == 'vggface2' else ThreatenedModelMNIST
    return model_cls(centroids_path=args.centroids_path, 
                    distance=args.distance, 
                    device=device, 
                    detector_ckp=args.detector_ckp, 
                    hidden=args.hidden, 
                    bidir=args.bidir, 
                    architecture=args.architecture, 
                    model_ckp=args.model_ckp)


def get_loader(dataset_name, imgs_dict):
    if dataset_name == 'vggface2':
        dataset = CustomDataset(root='/mnt/datone/datasets/vggface2/test-500', imgs_dict=imgs_dict)
        batch_size = 10
    else:
        dataset = MNIST(root='/media/fabiovalerio/datasets/mnist', transform=ToTensor(), train=False)
        batch_size = 100
    return DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=8)


def generate_targets(model, x, y, targeted, nb_classes):
    y = y.numpy()
    # Generate random target for targeted attacks
    if targeted:
        labels = np.random.randint(0, nb_classes, y.shape[0])
        indices = np.where(labels == y)[0]
        if len(indices) != 0:
            for i in indices:
                labels[i] = (y[i]+1)%nb_classes
    else:
        # Use model predictions as correct outputs
        preds, _ = model(x.cuda())
        preds_max = np.amax(preds.detach().cpu().numpy(), axis=1, keepdims=True)
        targets = preds.detach().cpu().numpy() == preds_max
        labels = targets.argmax(axis=-1)
    return torch.from_numpy(labels)
        

def main(args):

    if args.dataset_name == 'vggface2':
        MEAN_VECTOR = np.asarray([91.4953, 103.8827, 131.0912])[np.newaxis, np.newaxis, :]
    else:
        MEAN_VECTOR = np.asarray([0.])

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_d = os.path.join(MAIN_OUTPUT, args.dataset_name, args.attack)
    if not os.path.exists(out_d):
        os.makedirs(out_d)
    if args.attack == 'deep':
        out_d = os.path.join(out_d, 'max_thres_'+str(args.max_thres))
        if not os.path.exists(out_d):
            os.makedirs(out_d)
    if args.attack == 'cw2':
        out_d = os.path.join(out_d, 'abort_early_'+str(args.abort_early))
        if not os.path.exists(out_d):
            os.makedirs(out_d)
    print(f"Adversarials will be saved here: {out_d}")

    model = get_threatened_model(args, device)
    model.eval()
    ## In order to backpropagate we need the detector in train()
    ## It does not impact on the generation since it has not batch norm or other layers
    ## with difference behaviour when in eval()
    if args.architecture == 'lstm': model._detector.train()

    ## Get 100 correctly identified natural images, i.e. 2 for each class, to use for white box attacks
    imgs_dict = {}
    if args.dataset_name == 'vggface2':
        df = pd.read_csv('../../detector_training/output_csv/data_splits/data_split_targeted.csv')
        for idx, row in df[(df['adv'] == 0) & ((df['split'] == 'test') | (df['split'] == 'val'))].iterrows():
            k = row['path'].split('/')[0]
            if k not in imgs_dict:
                imgs_dict[k] = []
            if len(imgs_dict[k]) == 2: continue
            p = os.path.join('/mnt/datone/datasets/vggface2/test-500', row['path'])
            if not os.path.exists(p):
                p = os.path.join('/mnt/datone/datasets/vggface2/validation-500', row['path'])
            imgs_dict[k].append(p)
        ## The real total number of images is 939

    loader = get_loader(args.dataset_name, imgs_dict)

    knn_clf = None
    if args.attack == 'deep':
        base = '/mnt/datone/adversarial_face_recognition_original/features_centroids/train-500-w-dropout/'
        a = h5py.File(os.path.join(base, 'vggface2_medoids.h5'), 'r')
        ds = a['avg_pool'][:]
        a.close()
        knn_clf = KNeighborsClassifier(n_neighbors=1, weights='distance').fit(ds, np.arange(0, 500, 1))
        param_dict = dict(max_thres=args.max_thres)
    else: # cw2
        param_dict = dict(confidence=args.confidence, search_steps=args.search_steps, max_steps=args.max_steps, abort_early=args.abort_early)

    classifier_adv_success = 0
    classifier_adv_success_detected = 0
    classifier_adv_success_not_detected = 0
    
    tot = 0
    tt = 0
    for idx, (x, y) in enumerate(tqdm(loader, total=len(loader), leave=False), 1):
        
        if args.dataset_name == 'vggface2':
            o = model._classifier(x.cuda())[-1]
        else:
            o = model._classifier(x.cuda())
        corr = torch.where(o.max(-1)[1].cpu() == y)[0]
        x = x[corr]
        y = y[corr]

        _, detector_out = model(x.cuda())
        ys = F.sigmoid(detector_out).squeeze()
        corr = torch.where(ys < 0.5)[0]
        x = x[corr]
        y = y[corr]

        if args.attack == 'deep':
            # Consider only images correctly classified by the kNN
            feat_ = model.extract_features(x.to(device))
            pred = knn_clf.predict(feat_.detach().cpu().numpy())
            corr_pred_idx = np.where(pred == y.numpy())[0]
            x = x[corr_pred_idx]
            y = y[corr_pred_idx]
        
        targets = generate_targets(model, x, y, args.targeted, nb_classes=500 if args.dataset_name == 'vggface2' else 10)
        if args.attack == 'cw2' and not args.targeted:
            targets = y

        tot += x.shape[0]

        if targets.shape[0] == 0: continue

        x_adv = attack(dataset_name=args.dataset_name,
                    model=model, 
                    x=x, 
                    targets=targets, 
                    param_dict=param_dict, 
                    targeted=args.targeted, 
                    device=device,
                    atk=args.attack, 
                    knn_clf=knn_clf, 
                    guide_features=ds[targets] if args.attack == 'deep' else None)
        
        with torch.no_grad():
            classifier_out, detector_out = model(x.cuda())
            classifier_out_adv, detector_out_adv = model(x_adv.float().cuda())

        if args.attack == 'cw2':
            if args.targeted:
                misclassifier_adv = torch.where(classifier_out_adv.max(-1)[1].cpu() == targets)[0]
            else:
                misclassifier_adv = torch.where(classifier_out_adv.max(-1)[1].cpu() != y)[0]
        else:
            x_feat = model.extract_features(x_adv.cuda())
            pred_ = knn_clf.predict(x_feat.detach().cpu().numpy())
            if args.targeted:
                misclassifier_adv = np.where(pred_ == targets.numpy())[0]
            else:
                misclassifier_adv = np.where(pred_ != y.numpy())[0]

        classifier_adv_success += len(misclassifier_adv)

        ys = F.sigmoid(detector_out_adv).squeeze()
        ys = ys
        ys[ys >= 0.5] = 1
        ys[ys < 0.5] = 0

        classifier_adv_success_detected += len(torch.where(ys[misclassifier_adv] == 1)[0])
        classifier_adv_success_not_detected += len(torch.where(ys[misclassifier_adv] == 0)[0])

        if args.dataset_name == 'mnist' and not args.all_mnist:
            for ij_ in misclassifier_adv:
                tt += 1
                f_name = f"evasion_{tt}-org_class_{y[ij_].item()}_adv_pred_{classifier_out_adv.max(-1)[1][ij_].item()}_adv_target_{targets[ij_]}"
                f_name_nat = f"evasion_{tt}-org_class_{y[ij_].item()}_adv_pred_{classifier_out_adv.max(-1)[1][ij_].item()}_adv_target_{targets[ij_]}_nat"
                path = os.path.join(out_d, f_name)
                path_nat = os.path.join(out_d, f_name_nat)
                img_ = x_adv[ij_].squeeze().detach().cpu().numpy()
                np.savez_compressed(path, img=img_)
                img_ = x[ij_].squeeze().detach().cpu().numpy()
                np.savez_compressed(path_nat, img=img_)
        
        elif args.dataset_name == 'vggface2':
            for ij_ in torch.where(ys[misclassifier_adv] == 0)[0]:
                tt += 1
                f_name = f"evasion_{tt}-org_class_{y[ij_].item()}_adv_pred_{pred_[ij_]}_adv_target_{targets[ij_]}"
                f_name_nat = f"evasion_{tt}-org_class_{y[ij_].item()}_adv_pred_{pred_[ij_]}_adv_target_{targets[ij_]}_nat"
                path = os.path.join(out_d, f_name)
                path_nat = os.path.join(out_d, f_name_nat)
                img_ = x_adv[ij_].squeeze().detach().cpu().numpy()
                if args.dataset_name == 'vggface2':
                    img_ = np.transpose(img_, (1, 2, 0))
                img_ += MEAN_VECTOR
                np.savez_compressed(path, img=img_)
                img_ = x[ij_].squeeze().detach().cpu().numpy()
                if args.dataset_name == 'vggface2':
                    img_ = np.transpose(img_, (1, 2, 0))
                img_ += MEAN_VECTOR
                np.savez_compressed(path_nat, img=img_)

        ## Consider only 1000 samples from MNIST
        if not args.all_mnist and tot >= 1000 and args.dataset_name == 'mnist': break
        
    print(f"Attack: {args.attack}")
    print(f"Adversarials success rate:  {(classifier_adv_success/tot)*100:4.2f}%")
    print(f"Adversarial detection rate: {(classifier_adv_success_detected/tot)*100:4.2f}%")
    print(f"Adversarial evasion rate:   {(classifier_adv_success_not_detected/tot)*100:4.2f}%")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='white box')
    parser.add_argument('-s', '--seed', type=int, default=13)
    parser.add_argument('-ds', '--dataset-name', choices=('mnist', 'vggface2'), default='mnist')
    parser.add_argument('-alm', '--all-mnist', action='store_true')
    parser.add_argument('-dck', '--detector-ckp', help='Detector checkpoint')
    parser.add_argument('-mck', '--model-ckp', help='Model checkpoint')
    parser.add_argument('-cp', '--centroids-path', help='Path to class representatives')
    ## Attacks
    parser.add_argument('-atk', '--attack', choices=('cw2', 'deep'), default='deep')
    parser.add_argument('-t', '--targeted', action='store_true', help='Run targeted attacks (default: False)')
    ## Deep Features attacks
    parser.add_argument('-mt', '--max-thres', type=int, default=10)
    ## Classification attacks
    # CW2
    parser.add_argument('-c', '--confidence', type=float, default=0.0)
    parser.add_argument('-bs', '--search_steps', type=int, default=10)
    parser.add_argument('-ms', '--max-steps', type=int, default=1000)
    parser.add_argument('-al', '--abort-early', action='store_true')
    
    args = parser.parse_args()

    if args.dataset_name == 'vggface2':
        args.hidden = 100
        args.bidir = True
        args.architecture = args.detector_ckp.split('/')[-1].split('_')[4]
        args.distance = args.detector_ckp.split('/')[-1].split('_')[8]
    
    else:
        args.hidden = 100
        args.bidir = True
        args.architecture = 'lstm'
        args.distance = 'euclidean'
                            
    main(args)
