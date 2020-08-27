from __future__ import division

import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import foolbox

import torch
from torch.utils.data import DataLoader
from utils import *


def get_attack(attack, model, criterion, y, device):
    preproc = ((0, 0, 0), (1, 1, 1))
    preproc = map(lambda x: np.array(x).reshape((3, 1, 1)), preproc)  # expand dimensions
    preproc = tuple(preproc)
    fmodel = foolbox.models.PyTorchModel(model=model,
                                         bounds=(0, 1),
                                         preprocessing=preproc,
                                         num_classes=10,
                                         device=device)
    
    if attack == 'cw2':
        attack_class, distance = foolbox.attacks.CarliniWagnerL2Attack, foolbox.distances.MSE
    elif attack == 'bim2':
        attack_class, distance = foolbox.attacks.L2BasicIterativeAttack, foolbox.distances.MSE
    elif attack == 'pgd':
        attack_class, distance = foolbox.attacks.RandomStartProjectedGradientDescentAttack, foolbox.distances.Linf
    else: # attack == 'mi-fgsm':
        attack_class, distance = foolbox.attacks.MomentumIterativeAttack, foolbox.distances.Linf
    
    if criterion == 't':
        nb_classes = 10
        labels = np.random.randint(0, nb_classes, y.shape[0])
        indices = np.where(labels == y)[0]
        if len(indices) != 0:
            for i in indices:
                labels[i] = (y[i]+1)%nb_classes
        criterion_ = [foolbox.criteria.TargetClass(label) for label in labels]
    else:
        criterion_ = foolbox.criteria.Misclassification()

    return attack_class(model=fmodel, criterion=criterion_, distance=distance)


def get_attack_params(attack_name, args):
    if attack_name == 'cw2':
        return dict(binary_search_steps=args.binary_search_steps, max_iterations=args.max_iterations, 
                    confidence=args.confidence, learning_rate=0.01, initial_const=0.01, abort_early=True)
    elif attack_name == 'bim2':
        return dict(binary_search=False, epsilon=args.epsilon, stepsize=args.stepsize, 
                    iterations=args.max_iterations, random_start=False, return_early=True)
    elif attack_name == 'pgd':
        return dict(binary_search=False, epsilon=args.epsilon, stepsize=args.stepsize, 
                    iterations=args.max_iterations, random_start=False, return_early=True)
    elif attack_name == 'mi-fgsm':
        return dict(binary_search=False, epsilon=args.epsilon, stepsize=args.stepsize, 
                    iterations=args.max_iterations, decay_factor=1.0, random_start=False, 
                    return_early=True)


def get_out_main_path(attack_name, criterion, args):
    if attack_name == 'cw2':
        return f"{attack_name}-i_{args.max_iterations}-bs_{args.binary_search_steps}-confidence_{args.confidence}-targeted-{criterion}"
    else:
        return f"{attack_name}-eps_{args.epsilon}-epsi_{args.stepsize:.4f}-i_{args.max_iterations}-targeted-{criterion}"


def main(args):
    base_output = '/media/fabiovalerio/adversarial_face_recognition_revision_outputs/adversarials/'
    datasets_base_path = '/media/fabiovalerio/datasets'

    model = load_model(args.model_arc, args.model_checkpoint)

    loader = get_loader_for_adversarials_generation(datasets_base_path, args.dataset_name)
    batch_size = loader.batch_size

    save_nat_images = True
    nat_images_out_path = os.path.join(base_output, args.dataset_name, 'nat_images')
    if not os.path.exists(nat_images_out_path):
        os.makedirs(nat_images_out_path)

    attack_params = get_attack_params(args.attack_name, args)

    attack_output_folder_ = get_out_main_path(args.attack_name, args.criterion, args)
    attack_output_folder = os.path.join(base_output, args.dataset_name, attack_output_folder_)
    
    if not os.path.exists(attack_output_folder):
        os.makedirs(attack_output_folder)
    elif len(attack_output_folder) != 0:
        #print("Attack already run... skip!!!")
        #continue
        pass

    results_file_path = os.path.join(base_output, args.dataset_name, attack_output_folder_+'.csv')
    if not os.path.exists(results_file_path):
        results = pd.DataFrame()
    else:
        results = pd.read_csv(results_file_path)

    nat_file_path = os.path.join(base_output, args.dataset_name, 'nat.csv')
    if not os.path.exists(nat_file_path):
        nat_img_df = pd.DataFrame()
    else:
        nat_img_df = pd.read_csv(nat_file_path)
    
    successes = 0
    tot = 0
    n = 1
    nat = 1
    for idx, (x, y) in enumerate(tqdm(loader, desc=attack_output_folder_, leave=False), 1):
        logits = model(x.cuda())
        corr = torch.where(logits.max(-1)[1].cpu() == y)[0]
        x = x[corr]
        y = y[corr]
        tot += x.shape[0]

        attack = get_attack(args.attack_name, model, args.criterion, y.numpy(), args.device)
        adversarials = attack(x.numpy(), y.numpy(), unpack=False, **attack_params)
        
        for i_, adv in enumerate(adversarials):
            if adv.adversarial_class is not None:
                successes += 1
                path = os.path.join(attack_output_folder, f"id-{n}_org-class-{y[i_].item()}_adv_class-{adv.adversarial_class}")
                result = pd.DataFrame(dict(sample_id=n,
                                    original_class=y[i_].item(),
                                    adversarial_class=adv.adversarial_class,
                                    attack_name=args.attack_name,
                                    criterion=args.criterion,
                                    path=path,
                                ), index=[0]
                            )
                results = results.append(result, ignore_index=True)
                
                np.savez_compressed(path, img=np.transpose(adv.perturbed, (1, 2, 0)))
                n += 1

        results.to_csv(results_file_path, index=False)

        if save_nat_images:
            for ii, img_ in enumerate(x):
                path = os.path.join(nat_images_out_path, f"nat-{nat}_class-{y[ii].item()}")
                result = pd.DataFrame(dict(sample_id=nat,
                                    original_class=y[ii].item(),
                                    adversarial_class=None,
                                    attack_name='natural',
                                    criterion=None,
                                    path=path,
                                ), index=[0]
                            )
                nat_img_df = nat_img_df.append(result, ignore_index=True)

                np.savez_compressed(path, img=np.transpose(img_.detach().cpu().numpy(), (1, 2, 0)))
                nat += 1

            nat_img_df.to_csv(nat_file_path, index=False)

    print(f"Attack succes rate: {attack_output_folder_} - {(successes/tot)*100:4.2f}%")
    save_nat_images = False
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Attacks')

    parser.add_argument('-an', '--attack-name',  choices=('pgd', 'mi-fgsm', 'bim2', 'cw2'), default='pgd', help='Attack name (default: pgd)')
    parser.add_argument('-cr', '--criterion', choices=('t', 'ut'), default='ut', help='Attack criterion (default: ut')
    parser.add_argument('-e', '--epsilon',  type=float, default=0.3, help='maximum perturbation allowed')
    parser.add_argument('-ni', '--max-iterations', type=int, default=10, help='Number of iterations (default: 10)')
    parser.add_argument('-c', '--confidence', type=float, default=0.0, help='CW confidence (default: 0.0)')
    parser.add_argument('-bs', '--binary-search-steps', type=int, default=10, help='CW binary search steps (default: 10)')
    parser.add_argument('-ma', '--model-arc', choices=('small', 'wide'), default='small')

    args = parser.parse_args()

    args.stepsize = (args.epsilon / args.max_iterations) * 2
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model_arc == 'small':
        args.dataset_name = 'mnist'
        args.model_checkpoint = 'models/mnist/model-smallcnn.pth'
    else:
        args.dataset_name = 'cifar10'
        args.model_checkpoint = 'models/cifar10/model-wideres.pth'

    main(args)
