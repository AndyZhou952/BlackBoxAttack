from tqdm import tqdm 
import torch
import torch.nn.functional as F
from torch import nn


#Query limited attack using NES
def NES(model, target_class, image, search_var, sample_num, g, u):
    #parameters
    n = sample_num #should be even
    N = image.size(2) #assume the image is N x N may subject to change
    
    #NES estimation
    g.zero_()
    with torch.no_grad():
        for i in range(n):
            u.normal_()
            g = g + F.softmax(model(image + search_var * u_i), dim =1)[0,target_class] * u
            g = g - F.softmax(model(image - search_var * u_i), dim =1)[0,target_class] * u #we assume the output of the model is ordered by class index
    return 1 / (2*n*search_var) * g

def adversarial_generator(model, target_class, image, search_var, sample_num, bound, lr, query_limit):
    device = next(model.parameters()).device
    image = image.to(device)
    adv_image = image.clone()
    adv_image = adv_image.to(device)
    N = image.size(2)
    g = torch.zeros(N, requires_grad=False).to(device)
    u_i = torch.randn((N,N)).to(device)
    with torch.no_grad():
        for i in tqdm(range(query_limit // sample_num)):
            gradient = NES(model, target_class, adv_image, search_var, sample_num, g, u_i)
            tmp = adv_image - lr * torch.sign(gradient)
            adv_image = torch.clamp(tmp, min=image-bound, max=image + bound)
            adv_image = torch.clamp(adv_image, 0, 1)
    return adv_image