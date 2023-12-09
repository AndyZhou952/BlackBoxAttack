import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


def logit_margin(z, y):
    z_clone = z.clone().detach()
    z_clone.requires_grad = False
    fy = z[:, y]
    z_clone[:, y]= -1000
    top_logits = torch.topk(z_clone, dim=1, k=1).values[:, 0]
    return fy - top_logits

# z: logits output from model
# y: class label. If defender knows the true label, then use true label; otherwise, use prediction as label in loss
# T: temperature, a hyper-parameter to tune. Pre-tune it using validation set.
# alpha: magnitude of distortion
# beta: trade-off parameter for denfense & confidence preservation
# kappa: number of iterations for optimziation
# AAA_type: 'sin' or 'lin'. See paper
def AAA(z, y, alpha, tau, kappa, T, beta, lr, AAA_type='sin'):    
    # z: (batch_size, K)
    # y: (batch_size, 1), integer encoding
    device = z.device
    z = z.detach()
    y = y.detach()
    
    l_org = logit_margin(z, y)
    l_atr = (torch.floor(l_org / tau) + 1/2) * tau
    if AAA_type == 'sin': 
        l_trg = l_org - alpha *  tau * torch.sin(torch.pi * ( 1 - 2 * (l_org - l_atr) / tau))
    else:
        l_trg = l_atr - alpha * (l_org - l_atr)
    p_trg = torch.max(F.softmax(z / T, dim=1), dim=1).values
    
    u = z.clone().detach()
    u.requires_grad = True
    u = u.to(device)
    
    with torch.enable_grad():
        optimizer = torch.optim.Adam([u], lr=lr, betas=(0.9, 0.999))

        for i in range(kappa):
            optimizer.zero_grad()
            loss = torch.abs(logit_margin(u, y) - l_trg) + beta * torch.abs(torch.max(F.softmax(u, dim=1))- p_trg)
            loss = loss.sum()
            loss.backward()
            optimizer.step()
    return u


class AAAProtectedClassifier(nn.Module):
    def __init__(self, model,  alpha, tau, kappa, T, beta, lr, AAA_type='sin'):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.tau = tau
        self.kappa = kappa 
        self.T = T
        self.beta = beta
        self.lr = lr
        self.AAA_type = AAA_type
        
    def forward(self, x):
        # x: image batch of dimension (batch_size, C, H, W)
        x = x.to(device = next(self.model.parameters()).device)
        self.model.eval()
        with torch.no_grad():
            org_logits = self.model(x)
            pred_y = org_logits.argmax(dim=1)
        
        protected_logits = AAA(z=org_logits, y=pred_y,
                               alpha=self.alpha,
                               tau=self.tau,
                               kappa=self.kappa,
                               T=self.T,
                               beta=self.beta,
                               lr=self.lr,
                               AAA_type=self.AAA_type)
        
        return protected_logits
    
    
    
class PartialInfo(nn.Module):
    def __init__(self, model,  k):
        super().__init__()
        self.model = model
        self.k = k
        
    def forward(self, x):
        # x: image batch of dimension (batch_size, C, H, W)
        x = x.to(device = next(self.model.parameters()).device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
        # threshold: (batch_size, 1)
        threshold = torch.topk(logits, self.k, dim=1).values[:, self.k-1]
        # threshold: (batch_size, 100)
        threshold = threshold.unsqueeze(1).expand_as(logits)
        logits[logits < threshold] = float('nan')
        
        return logits