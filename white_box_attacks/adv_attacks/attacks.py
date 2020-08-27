import sys
sys.path.append('/home/fabiovalerio/lavoro/adversarial_face_recognition/paper-revision/white_box_attacks_face')

import numpy as np

from adv_attacks.attack_cw2 import L2Adversary as cw2
from adv_attacks.attack_deep_feat import deep_features_attack


def attack(dataset_name, model, x, targets, param_dict, targeted, device, atk, knn_clf, guide_features):
    if atk == 'deep' and dataset_name == 'vggface2': ## only for VGGFace2 
        MEAN_VECTOR = np.asarray([91.4953, 103.8827, 131.0912])[:, np.newaxis, np.newaxis]
        MIN = -MEAN_VECTOR
        MAX = -MEAN_VECTOR+255.
        return deep_features_attack(model, x, targets, device, knn_clf, targeted, (MIN, MAX), param_dict['max_thres'], guide_features)

    elif atk == 'cw2' and dataset_name == 'mnist': ## only for MNIST
        adversary = cw2(num_classes=10, targeted=targeted, confidence=param_dict['confidence'], 
                        search_steps=param_dict['search_steps'], box=(0, 1), max_steps=param_dict['max_steps'], 
                        abort_early=param_dict['abort_early'], optimizer_lr=5e-4)
        return adversary(model, x.cuda(), targets.cuda(), to_numpy=False)

    else:
        print("Attack - Dataset combination not valid")
        sys.exit(1)
