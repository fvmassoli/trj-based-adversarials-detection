import os
import cv2
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from utils import load_model, get_transforms, TinyDataset


def extract_features(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = load_model(base_model_path=args.base_model_path, model_checkpoint=args.model_checkpoint, device=device)

    out_folder = args.features_folder
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for n_ in tqdm(os.listdir(args.dataset_path), total=len(os.listdir(args.dataset_path)), desc='Extracting features'):
        dataloader = DataLoader(dataset=TinyDataset(root=os.path.join(args.dataset_path, n_), 
                            transform=get_transforms()), 
                            batch_size=args.batch_size, 
                            pin_memory=torch.cuda.is_available(), 
                            num_workers=8)

        nb_images = len(dataloader.dataset)
        n_processed = 0

        with torch.no_grad():
            ff = h5py.File(os.path.join(out_folder, n_+'.h5'))

            for x in dataloader:
                out = model(x.to(device))

                for k in out.keys():
                    extracted = out[k]

                    if k != 'avg_pool' and k != 'classifier' and extracted.ndimension() > 2:
                        extracted = F.avg_pool2d(out[k], out[k].shape[-2:]).squeeze(3).squeeze(2)

                    batch_size, feature_dims = extracted.shape
                    if batch_size != out[k].shape[0]: print('error', batch_size, out[k].shape)
                    dset = ff.require_dataset(k, (nb_images, feature_dims), dtype='float32', chunks=(50, feature_dims))
                    dset[n_processed:n_processed + batch_size, :] = extracted.to('cpu')

                n_processed += batch_size

            ff.close()

        del dataset
        del dataloader


def compute_class_representatives(args):
    features_files = [i for i in os.listdir(args.features_folder) if i.endswith('.h5')]
    features_files.sort()
    features_files = [os.path.join(args.features_folder, i) for i in features_files]

    with h5py.File(features_files[0], 'r') as features:
        layer_names = sorted(features.keys())

    class_centroids = []
    for class_features in tqdm(features_files, total=len(features_files), desc='Computing class representatives'):
        with h5py.File(class_features, 'r') as features:
            centroids = [compute_centroid(features[layer], medoid=args.medoid) for layer in tqdm(layer_names)]
            class_centroids.append(centroids)

    layer_centroids = zip(*class_centroids)

    print('Saving centroids:', args.output_file)
    with h5py.File(args.output_file, 'w') as out:
        for layer, centroids in tqdm(zip(layer_names, layer_centroids)):
            centroids = np.stack(centroids)
            out.create_dataset('{}'.format(layer), data=centroids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Features extraction with pre-trained CNN')
    parser.add_argument('-dp', '--dataset-path', help='Load Model Checkpoint')
    parser.add_argument('-bs', '--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('-bm', '--base-model-path', help='Path to base model checkpoint')
    parser.add_argument('-ck', '--model-checkpoint', help='Fine-tuned model checkpoint path')
    parser.add_argument('-ff', '--features-folder', help='Output folder')
    parser.add_argument('-cr', '--class-representatices', choices=('c', 'm'), default='c', 
                    help='Compute class representatives (default: centroids).')
    parser.add_argument('-d', '--distance', default='euclidean',
                    help='Distance metric for computing medoids (see scipy.spatial.distance.pdist for choices).')
    args = parser.parse_args()

    if args.extract_features:
        print("*"*20, 'Features Extraction', "*"*20)
        extract_features(args)
        print("*"*20, 'Evaluating Class Representatives', "*"*20)
        compute_class_representatives(args)
    else:
        print("*"*20, 'Evaluating Class Representatives', "*"*20)
        compute_class_representatives(args)
