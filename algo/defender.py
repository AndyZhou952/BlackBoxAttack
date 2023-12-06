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
    # z: (batch_size, K)
    # y: (batch_size, 1), integer encoding
    device = z.device
    z = z.detach()
    
    l_org = L(z, y)
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
        optimizer = torch.optim.AdamW([u], lr=lr)

        for i in range(kappa):
            optimizer.zero_grad()
            loss = torch.abs(L(u, y) - l_trg) + beta * torch.abs(torch.max(F.softmax(u, dim=1))- p_trg)
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
        self.L = nn.CrossEntropyLoss(reduce=False)
        
    def forward(self, x):
        # x: image batch of dimension (batch_size, C, H, W)
        x = x.to(device = next(self.model.parameters()).device)
        self.model.eval()
        with torch.no_grad():
            org_logits = self.model(x)
        pred_y = org_logits.argmax(dim=1)
        
        protected_logits = AAA(z=org_logits, y=pred_y, L=self.L,
                               alpha=self.alpha,
                               tau=self.tau,
                               kappa=self.kappa,
                               T=self.T,
                               beta=self.beta,
                               lr=self.lr,
                               AAA_type=self.AAA_type)
        
        return protected_logits