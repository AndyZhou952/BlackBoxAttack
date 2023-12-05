import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

# z: logits output from model
# y: class label. If defender knows the true label, then use true label; otherwise, use prediction as label in loss
# T: temperature, a hyper-parameter to tune. Pre-tune it using validation set.
# L: model loss function. For multi-class classification, it is nn.CrossEntropyLoss()
# alpha: magnitude of distortion
# beta: trade-off parameter for denfense & confidence preservation
# kappa: number of iterations for optimziation
# AAA_type: 'sin' or 'lin'. See paper
def AAA(z, y, L, alpha, tau, kappa, T, beta, lr, AAA_type='sin'):    
    l_org = L(z, y).item()
    l_atr = (np.floor(l_org / tau) + 1/2) * tau
    if AAA_type == 'sin': 
        l_trg = l_org - alpha *  tau * np.sin(np.pi * ( 1 - 2 * (l_org - l_atr) / tau))
    else:
        l_trg = l_atr - alpha * (l_org - l_atr)
    p_trg = torch.max(F.softmax(z / T, dim=1)).item()

    
    u = z.clone().detach()
    u.requires_grad = True
    u = u.to(z.device)
    
    with torch.enable_grad():
        optimizer = torch.optim.AdamW([u], lr=lr)

        for i in range(kappa):
            optimizer.zero_grad()
            loss = torch.abs(L(u, y) - l_trg) + beta * torch.abs(torch.max(F.softmax(u, dim=1))- p_trg)
            loss.backward()
            optimizer.step()
    
    return u