import sys
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

from scipy.optimize import fmin_l_bfgs_b


def calc_gstep(x, model, orig_feat, y, guide_features, guide_label, knn_clf, targeted, device):
    x = torch.tensor(x.reshape((3, 224, 224)), requires_grad=True, dtype=torch.float32)
    x_feat = model.extract_features(x[np.newaxis, :].to(device))
    
    if targeted:
        is_adv = knn_clf.predict(x_feat.detach().cpu().numpy()[np.newaxis, :])[0] == guide_label
        loss_a = torch.norm(x_feat - torch.from_numpy(guide_features).to(device))
    else:
        is_adv = knn_clf.predict(x_feat.detach().cpu().numpy()[np.newaxis, :])[0] != y
        raise NotImplemented()
        
    _, det_out = model(x[np.newaxis, :].to(device))
    loss_b = F.binary_cross_entropy_with_logits(det_out[0], torch.tensor([0]).float().cuda())
    loss = loss_a + (10.0*loss_b)
    
    if is_adv:
        loss = 0.0
        grad = torch.zeros_like(x)
    else:
        loss.backward()
        grad = x.grad.data
    
    return loss, grad.detach().cpu().numpy().flatten().astype(float)


def deep_features_attack(model, x, y, device, knn_clf, targeted, box, max_thres, guide_features):
    factr = 10000000.0
    pgtol = 1e-05
    iter_n = 1000
    
    x_adv = []

    for src_, original_class, guide_feature in zip(x, y, guide_features):
        
        up_bnd = src_ + max_thres
        lw_bnd = src_ - max_thres
        
        mean_arr = np.asarray([91.4953, 103.8827, 131.0912])
        if mean_arr.ndim == 1:
            mean_arr = mean_arr.reshape((3, 1, 1))
            
        up_bnd = np.minimum(up_bnd, 255 - mean_arr)
        lw_bnd = np.maximum(lw_bnd, 0 - mean_arr)
        
        bound = list(zip(lw_bnd.flatten(), up_bnd.flatten()))
        
        assert (up_bnd-lw_bnd > 2*max_thres + 1.e-3).sum() == 0
        assert (up_bnd-lw_bnd < 0).sum() == 0
        
        xx, f, d = fmin_l_bfgs_b(func=calc_gstep,
                        x0=src_.numpy().flatten(),
                        args=(model,
                            model.extract_features(src_.cuda()[np.newaxis, :]).detach().cpu().numpy(),
                            original_class.numpy(), 
                            guide_feature,
                            knn_clf.predict(guide_feature[np.newaxis, :])[0], ## guide label
                            knn_clf,
                            targeted,
                            device),
                        bounds=bound,
                        maxiter=iter_n,
                        iprint=0,
                        factr=factr,
                        pgtol=pgtol)
        
        adv = torch.tensor(xx.reshape(3, 224, 224), dtype=torch.float32).to(device)
        pred = knn_clf.predict(model.extract_features(adv[np.newaxis, :]).detach().cpu().numpy()[np.newaxis, :])[0]
        
        x_adv.append(adv)
    
    return torch.stack([x for x in x_adv])

