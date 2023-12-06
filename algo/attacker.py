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
            g = g + F.softmax(model(image + search_var * u), dim =1)[0,target_class] * u
            g = g - F.softmax(model(image - search_var * u), dim =1)[0,target_class] * u #we assume the output of the model is ordered by class index
    return 1 / (2*n*search_var) * g

def adversarial_generator(model, target_class, image, search_var, sample_num, bound, lr, query_limit):
    device = next(model.parameters()).device
    image = image.to(device)
    adv_image = image.clone()
    adv_image = adv_image.to(device)
    N = image.size(2)
    g = torch.zeros(N, requires_grad=False).to(device)
    u = torch.randn((N,N)).to(device)
    with torch.no_grad():
        for i in tqdm(range(query_limit // sample_num)):
            gradient = NES(model, target_class, adv_image, search_var, sample_num, g, u)
            tmp = adv_image - lr * torch.sign(gradient)
            adv_image = torch.clamp(tmp, min=image-bound, max=image + bound)
            adv_image = torch.clamp(adv_image, 0, 1)
    return adv_image

# PIA algo
def PIA_adversarial_generator(model, initial_image, image, target_class, epsilon_adv, epsilon_0, search_var, sample_num, delta_epsilon, eta_max, eta_min, k=5):
    device = next(model.parameters()).device
    initial_image, image = initial_image.to(device), image.to(device)
    x_adv = image.clone().to(device)
    N = initial_image.size(2)
    g = torch.zeros(N, requires_grad=False).to(device)
    u = torch.randn((N, N)).to(device)
    
    epsilon = epsilon_0
    x_adv = torch.clamp(x_adv, initial_image - epsilon, initial_image + epsilon)
    original_class = torch.argmax(model(initial_image), dim=1)
    
    with torch.no_grad():
        probabilities_adv = F.softmax(model(x_adv), dim=1)
        top_probs, top_classes = torch.topk(probabilities, k)
        while (epsilon > epsilon_adv) | (original_class != target_class):
            gradient = NES(model, target_class, x_adv, search_var, sample_num, g, u)
            eta = eta_max
            x_adv_hat = x_adv - eta * gradient
            
            while not target_class in top_classes[0]:
                if eta < eta_min:
                    epsilon += delta_epsilon
                    delta_epsilon /= 2
                    x_adv_hat = x_adv
                    break  
                eta /= 2
                x_adv_hat = torch.clamp(x_adv_hat, initial_image - epsilon, initial_image + epsilon)
            x_adv = x_adv_hat
            epsilon = epsilon - delta_epsilon
    return x_adv

# Label-only attacker
