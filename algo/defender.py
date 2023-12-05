import torch
import torch.nn.functional as F
from torch import nn

# z: logits output from model
# y: class label. If defender knows the true label, then use true label; otherwise, use prediction as label in loss
# T: temperature, a hyper-parameter to tune. Pre-tune it using validation set.
# L: model loss function. For multi-class classification, it is nn.CrossEntropyLoss()
# alpha: magnitude of distortion
# beta: trade-off parameter for denfense & confidence preservation
# kappa: number of iterations for optimziation
# AAA_type: 'sin' or 'lin'. See paper
def AAA(z, y, L, alpha, tau, kappa, T, beta, AAA_type='sin'):
    l_org = L(z, y).item()
    l_atr = (torch.floor(l_org / tau) + 1/2) * tau
    if AAA_type = 'sin': 
        l_trg = l_org - alpha *  tau * torch.sin(torch.pi * ( 1 - 2 * (l_org - l_atr) / tau))
    else:
        l_trg = l_atr - alpha * (l_org - l_atr)
    
    l1 = nn.L1Loss()
    p_trg = torch.max(F.softmax(z / T))
    def optim_loss(u):
        return l1(L(u, y), l_trg).item() + beta * l1(torch.max(F.softmax(u)), p_trg).item()
    
    
    u = z.clone()
    optimizer = torch.optim.AdamW([u], lr=0.01)
    for i in range(kappa):
        optimizer.zero_grad()
        loss = optim_loss(u)
        loss.backward()
        optimizer.step()
    
    return u