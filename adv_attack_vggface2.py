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
from utils import load_model, get_adv_transforms, CustomDatasetFromCsv


def get_attack_class_and_distance(attack_class):
    if attack_class == 'cw2':
        return foolbox.attacks.CarliniWagnerL2Attack, foolbox.distances.MSE
    elif attack_class == 'bim2':
        return foolbox.attacks.L2BasicIterativeAttack, foolbox.distances.MSE
    elif attack_class == 'biminf': # same as PGD
        return foolbox.attacks.LinfinityBasicIterativeAttack, foolbox.distances.Linf
    elif attack_class == 'mi-fgsm':
        return foolbox.attacks.MomentumIterativeAttack, foolbox.distances.Linf
    else:
        raise NameError('Attack class not valid. Possible choices are: cw2, bim2, biminf')


def get_criterion(criterion):
    if criterion == 't':
        nb_classes = 500
        labels = np.random.randint(0, nb_classes, y.shape[0])
        indices = np.where(labels == y)[0]
        if len(indices) != 0:
            for i in indices:
                labels[i] = (y[i]+1)%nb_classes
        return [foolbox.criteria.TargetClass(label) for label in labels]
    else:
        return foolbox.criteria.Misclassification()
    

def main(args):
    device = 'cuda' if torhc.cuda.is_available() else 'cpu'

    model = load_model(base_model_path=args.base_model_path, model_checkpoint=args.model_checkpoint, device=device)
    
    data = CustomDatasetFromCsv(root=args.data_root,
                                csv_root='./data_splits',
                                transform=get_adv_transforms(),
                                return_path=True)

    preproc = ((91.4953, 103.8827, 131.0912), (1, 1, 1))
    preproc = map(lambda x: np.array(x).reshape((3, 1, 1)), preproc)  # expand dimensions
    preproc = tuple(preproc)

    fmodel = foolbox.models.PyTorchModel(model=model,
                                        bounds=(0, 255),
                                        num_classes=500,
                                        preprocessing=preproc,
                                        device=device)

    attack_class, distance = get_attack_class_and_distance(args.attack)
    criterion = get_criterion(args.criterion)
    attack = attack_class(model=fmodel,
                        criterion=criterion,
                        distance=distance)

    model_name = args.model_checkpoint.split('/')[-1].split('.')[0]

    if args.attack == 'cw2':
        binary_search_steps = 1
        confidence = 0
        results_file_name = 'adv_attack_' + args.attack + '_criterion-' + args.criterion + '_bsstep-' + str(
            binary_search_steps) +'_conf-'+str(confidence)+'_dp-' + str(
            args.useDropOut) + '_mname-' + model_name + '.csv'
    else:
        nb_iterations = args.attack_iterations
        step_size = args.step_size
        results_file_name = 'adv_attack_' + args.attack + '_criterion-' + args.criterion + '_eps-' + str(
            args.epsilon) + '_niter-' + str(nb_iterations) + '_dp-' + str(
            args.useDropOut) + '_mname-' + model_name + '.csv'

    results_file_folder = os.path.join(args.adv_output_path, 'adversarial_output_csv')
    if not os.path.exists(results_file_folder):
        os.makedirs(results_file_folder)
    results_file_path = os.path.join(results_file_folder, results_file_name)
    print('\nAttacks results .csv file will be saved at: {}'.format(results_file_path))

    adversarial_images_folder = os.path.join(args.adv_output_path, 'adversarial_output_images')
    t_ = 'targeted' if args.criterion == 't' else 'untargeted'
    adversarial_images_folder = os.path.join(adversarial_images_folder, args.attack, , t_)
    if not os.path.exists(adversarial_images_folder):
        os.makedirs(adversarial_images_folder)
    print('\nAdversarial images will be saved at: {}'.format(adversarial_images_folder))

    results = pd.read_csv(results_file_path) if os.path.exists(results_file_path) else pd.DataFrame()

    if args.attack == 'cw2':
        print('\nAttacks will be run with the following params:'
              '\n\tattack:              {}'
              '\n\tcriterion:           {}'
              '\n\tdistance:            {}'
              '\n\tbinary_search_steps: {}'
              '\n\tconfidence:          {}\n'.format(attack_class, distance, criterion, binary_search_steps, confidence))
    else:
        print('\nAttacks will be run with the following params:'
              '\n\tattack:    {}'
              '\n\tcriterion: {}'
              '\n\tdistance:  {}'
              '\n\tepsilon:   {}'
              '\n\tstep size: {}\n'.format(attack, distance, criterion, args.epsilon, step_size))

    if args.attack == 'cw2':
        attack_args = dict(learning_rate=1e-2, binary_search_steps=binary_search_steps, confidence=confidence)
    elif args.attack == 'bim2':
        attack_args = dict(random_start=True, binary_search=False, stepsize=step_size, epsilon=args.epsilon)
    elif args.attack == 'biminf' or args.attack == 'mi-fgsm':
        attack_args = dict(binary_search=False, stepsize=step_size, epsilon=args.epsilon)
    else:
        raise NameError('Attack class not valid. Possible choices are: cw2, bim2, biminf, mi-fgsm')

    progress = tqdm(data)
    for i, (image, label, image_path) in enumerate(progress):

        start = time.time()
        adversarial = attack(image, label, unpack=False) # , **attack_args)
        elapsed = time.time() - start

        image_ = image - np.asarray([91.4953, 103.8827, 131.0912]).reshape(3, 1, 1).astype(np.float32)
        lo1 = model(torch.from_numpy(image_[np.newaxis, :]).to(args.device)).detach().cpu().numpy()
        lo2 = fmodel.forward(image[np.newaxis, :])

        if adversarial.adversarial_class is not None:
            path = os.path.join(adversarial_images_folder, image_path.split('/')[-2])

            if args.attack == 'biminf' or args.attack == 'mi-fgsm':
                path = os.path.join(adversarial_images_folder, 'eps_'+str(args.epsilon), 'niterations_'+str(nb_iterations), image_path.split('/')[-2])

            if not os.path.exists(path):
                os.makedirs(path)

            np.save(os.path.join(path, image_path.split('/')[-1].split('.')[0]), adversarial.unperturbed)
            np.save(os.path.join(path, 'adversarial_of_class_'+str(adversarial.adversarial_class)+'_from_image_'+image_path.split('/')[-1].split('.')[0]), adversarial.perturbed)

        result = pd.DataFrame(dict(sample_id=i,
                                   label=label,
                                   elapsed_time=elapsed,
                                   distance=adversarial.distance.value,
                                   adversarial_class=adversarial.adversarial_class,
                                   original_class=adversarial.original_class,
                                   model_predicted_class=np.argmax(lo1),
                                   fmodel_predicted_class=np.argmax(lo2),
                                   model_pred_same_as_fmodel_pred_org_image=np.all(lo1 == lo2)
                                   ), index=[0])

        results = results.append(result, ignore_index=True)
        results.to_csv(results_file_path, index=False)

        success = ~results.adversarial_class.isna()
        successes = success.sum()
        
        progress.set_postfix(success_rate='{:.2f}%'.format(((successes/len(success)))*100.))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Attacks')
    parser.add_argument('-dr', '--data-root', help='Path to images')
    parser.add_argument('-a', '--attack', choices=['cw2', 'bim2', 'biminf', 'mi-fgsm'], default='bim2', help='Attack to perform (default: bim2)')
    parser.add_argument('-c', '--criterion', choices=['t', 'ut'], default='ut', help='Criterion (default: untargeted)')
    parser.add_argument('-e', '--epsilon',  type=float, default=0.3, help='maximum perturbation allowed')
    parser.add_argument('-ni', '--attack-iterations', type=int, default=10, help='Number of attack iterations (default: 10)')
    parser.add_argument('-st', '--step-size', type=float, default=0.007, help='Step size (default: 0.007)')
    parser.add_argument('-bm', '--base-model-path', help='Path to base model checkpoint')
    parser.add_argument('-ck', '--model-checkpoint', help='Fine-tuned model checkpoint path')
    parser.add_argument('-o', '--adv-output-path', help='Adversarial output path')
    args = parser.parse_args()
    
    main(args)
