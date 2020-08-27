import sys
sys.path.append('./')

import os
import h5py
import argparse
import itertools
import numpy as np
from tqdm import tqdm

from utils import *

import torch.nn as nn
import torch.nn.functional as F


def extract_features(args):
    feat_d = {}

    model = load_model(args.model_arc, args.model_checkpoint)

    if args.model_arc == 'small':
        blocks = list(itertools.chain((model.feature_extractor.relu2,), (model.feature_extractor.relu3,), (model.feature_extractor.relu4,), (model.classifier.relu1,), (model.classifier.relu2,)))
    else:
        blocks_ = list(itertools.chain(model.block1.layer, model.block2.layer, model.block3.layer, (model.avg_pool,)))
        blocks = [block.relu2 for block in blocks_ if not isinstance(block, nn.AvgPool2d)]
        blocks.append(blocks_[-1])                   
    n_features = len(blocks)
    blocks_idx = dict(zip(blocks, map('{:02d}'.format, range(n_features))))

    def hook_func(module, input, output):
        block_num = blocks_idx[module]
        extracted = output.detach()
        if extracted.ndimension() > 2:
            extracted = F.avg_pool2d(extracted, extracted.shape[-2:])
        feat_d[block_num] = extracted

    registered_hooks = [b.register_forward_hook(hook_func) for b in blocks_idx] 
    
    train_classes = np.arange(10)
    loader = get_loader_for_features_extraction('/media/fabiovalerio/datasets', args.dataset_name)

    ## Remove old features files
    if len(os.listdir(args.features_path)) != 0:
        print(f"Removing features files at: {args.features_path}")
        [os.remove(os.path.join(args.features_path, f)) for f in os.listdir(args.features_path)]
    ## Init output features files
    features_class_dict = {cls_: h5py.File(os.path.join(args.features_path, 'feat_cls_'+str(cls_)+'.hdf5'),'w-') for cls_ in train_classes}
    n_processed_d = {cls_: 0 for cls_ in train_classes}
    layer_names = sorted(list(blocks_idx.values()))
    ## Start features extraction
    for f_idx, (imgs, labels) in enumerate(tqdm(loader, total=len(loader), leave=False, desc=f"{args.model_arc}"), 1):
        logits = model(imgs.cuda())
        correctx_idx = torch.where(logits.max(-1)[1].cpu() == labels)[0]
        for label in train_classes:
            ## Get the file relative to the specific class
            f_ = features_class_dict[label]
            indices = np.where(labels[correctx_idx] == label)[0]
            ## For each class loop over all the layers from which we extracted the features
            for layer in layer_names:
                feature_dims = feat_d[layer].shape[1]
                if layer in f_:
                    dset = f_[layer]
                    dset.resize((dset.shape[0]+len(indices), feature_dims))
                else:
                    dset = f_.require_dataset(layer, (len(indices), feature_dims), maxshape=(None, feature_dims), dtype='float32')
                ## Save features
                dset[n_processed_d[label]:n_processed_d[label]+len(indices), :] = feat_d[layer][correctx_idx][indices].squeeze().detach().cpu()    
            ## Update indices
            n_processed_d[label] = n_processed_d[label] + len(indices)
    ## Close all hdf5 files
    [ff.close() for k, ff in features_class_dict.items()]


def eval_class_representatives(args):
    class_centroids = [] # each element is the list of centroids for a specfic class
    class_medoid_idx = [] 
    ffs_ = sorted(os.listdir(args.features_path))
    
    print(f"{args.model_arc}-{args.class_representatives}")

    for cls_file in ffs_:
        ## Open the file relative a specific class
        features_file = os.path.join(args.features_path, cls_file)
        print("\tWorking on file: {}".format(features_file))
      
        with h5py.File(features_file, 'r') as class_features:
            layer_names = sorted(class_features.keys())
            centroids_by_layer = []
            medoid_idx_by_layer = []

            ## For each layer, evaluate the centroid for the specific class
            for layer in layer_names:
                centroids_by_layer_, medoid_idx = compute_centroid(args.class_representatives, class_features[layer][:].squeeze())
                centroids_by_layer.append(centroids_by_layer_)
                medoid_idx_by_layer.append(medoid_idx)

            class_centroids.append(centroids_by_layer)
            class_medoid_idx.append(medoid_idx_by_layer)
    
    data = []
    for idx_, layer in enumerate(layer_names):
        data.append(np.stack([class_centroids[cls_][idx_] for cls_ in np.arange(len(class_centroids))]))
    
    f_name = 'centroids.hdf5' if args.class_representatives == 'c' else 'medoids.hdf5'
    centr_path = os.path.join(class_representatives_main_path, f_name)
    print('Saving representatives at: {}'.format(centr_path))
    print()
    with h5py.File(centr_path, 'w') as out:
        [out.create_dataset('{}'.format(layer), data=data[idx_]) for idx_, layer in enumerate(layer_names)]


if __name__ == '__main__':  
    parser = argparse.ArgumentParser('extr')
    parser.add_argument('-cr', '--class-representatives', choices=('c', 'm'), default='c')
    parser.add_argument('-ma', '--model-arc', choices=('small', 'wide'), default='small')
    parser.add_argument('-ex', '--extract', action='store_true')
    parser.add_argument('-ev', '--eval-centr', action='store_true')
    args = parser.parse_args()

    main_output_dir = '/media/fabiovalerio/adversarial_face_recognition_revision_outputs/'
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)

    if args.model_arc == 'small':
        args.dataset_name = 'mnist'
        args.model_checkpoint = 'models/mnist/model-smallcnn.pth'
    else:
        args.dataset_name = 'cifar10'
        args.model_checkpoint = 'models/cifar10/model-wideres.pth'

    feat_path = os.path.join(main_output_dir, args.dataset_name, 'features')
    if not os.path.exists(feat_path):
        os.makedirs(feat_path)

    class_representatives_main_path = os.path.join(main_output_dir, args.dataset_name, 'class_representatives')
    if not os.path.exists(class_representatives_main_path):
        os.makedirs(class_representatives_main_path)

    args.features_path = feat_path
    args.class_representatives_main_path = class_representatives_main_path

    if args.extract:
        extract_features(args)
    if args.eval_centr:
        eval_class_representatives(args)
