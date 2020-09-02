import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils import load_model, get_transforms, CustomDataset


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(base_model_path=args.base_model_path, model_checkpoint=args.model_checkpoint, device=device)

    dataset = CustomDataset(root=args.input_image_folder, transform=get_transforms(), return_paths=True)
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=32, pin_memory=True, num_workers=8)

    output_folder = './classification_output_files'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    label_file = open(os.path.join(output_folder, 'correctly_classified_images_'+args.output_filename+'.txt'), 'w')
    results_file_path = os.path.join(output_folder, 'classified_images_'+args.output_filename+'.csv')

    tot = 0
    correct_ = 0
    results = pd.DataFrame()

    progress_bar = tqdm(dataloader, total=len(dataloader), desc='Classifying: {}'.format(args.inputImageFolder.split('/')[-1]))

    for x, l, p in progress_bar:
        tot += x.shape[0]
        x = x.to(device)

        y_hat = model(x, True).max(-1)[1].cpu()
        correct = torch.where(y_hat == l)[0]
        correct_ += y_hat.eq(l).sum().item()

        progress_bar.set_postfix(accuracy=correct_)

        [label_file.write(p[idx] + '\n') for idx in correct]

        for i in range(len(l)):
            result = pd.DataFrame(dict(label=l[i].item(),
                                       predicted=y_hat[i].item()
                                       ), index=[0])

            results = results.append(result, ignore_index=True)
            results.to_csv(results_file_path, index=False)

    label_file.close()

    print("Accuracy: {:.2f}%".format(correct_/tot * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify test images')
    parser.add_argument('-i', '--input-image-folder', help='Input image folder')
    parser.add_argument('-o', '--output-filename', help='Output file name')
    parser.add_argument('-bm', '--base-model-path', help='Path to base model checkpoint')
    parser.add_argument('-ck', '--model-checkpoint', help='Fine-tuned model checkpoint path')
    args = parser.parse_args()

    main(args)
