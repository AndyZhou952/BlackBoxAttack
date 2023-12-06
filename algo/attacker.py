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

def get_rank_of_target(model, image, target_class, k = 5):
    """
    return the rank of the target in the top-k predictions
    return 0 if the target class is not in the top-k
    """
    device = next(model.parameters()).device
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
    probabilities = F.softmax(output, dim = 1)
    top_probs, top_classes = torch.topk(probabilities, k)
    if target_class in top_classes[0]:
        target_rank = (top_classes[0] == target_class).nonzero(as_tuple = True)[0].item() + 1
        return target_rank
    else:
        return 0

def S_x(model, image, target_class, mu, n, k):
    device = image.device
    R_sum = 0.0
    
    for _ in range(n):
        delta = (torch.rand_like(image) * 2 - 1) * mu
        perturbed_image = image + delta
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        if is_target_in_top_k(model, perturbed_image, target_class, k): # else R = 0, nothing added
            R = k - get_rank_of_target(model, perturbed_image, target_class, k)
            R_sum += R
    
    return R_sum / n    

def NES_label_only(model, target_class, image, search_var, sample_num, g, u, mu, k):
    n = sample_num
    N = image.size(2)

    g.zero_()
    with torch.no_grad():
        for _ in range(n):
            u.normal_()

            S_x_pos = S_x(model, image + search_var * u, target_class, mu, n, k)
            S_x_neg = S_x(model, image - search_var * u, target_class, mu, n, k)

            g += (S_x_pos - S_x_neg) * u

    return g / (2 * n * search_var)

def PIA_adversarial_generator(model,initial_image, target_image, target_class, epsilon, 
                              delta, search_var, sample_num, eta_max, eta_min, epsilon_adv, k, query_limit = 100000, 
                              label_only = False, mu = None):
    query_count = 0
    
    device = next(model.parameters()).device
    x = initial_image.to(device)
    x_adv = target_image.clone().to(device)
    x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
    N = initial_image.size(2)
    g = torch.zeros(N, requires_grad = False).to(device)
    u = torch.randn((N, N)).to(device)
    
    with torch.no_grad():
        while epsilon > epsilon_adv or not get_rank_of_target(model, x_adv, target_class, k = 1):
            if query_count >= query_limit:
                return False # indicating non_convergence
            
            if label_only:
                gradient = NES_label_only(model, target_clss, x_adv, search_var, sample_num, g, u, mu, k)
            else:
                gradient = NES(model, target_clss, x_adv, search_var, sample_num, g, u)
            eta = eta_max
            x_adv_hat = x_adv - eta * gradient
            
            while not get_rank_of_target(model, x_adv_hat, target_class, k):
                if eta < eta_min:
                    epsilon += delta
                    delta /= 2
                    x_adv_hat = x_adv
                    break
                eta /= 2
                x_adv_hat = torch.clamp(x_adv - eta * gradient, x - epsilon, x + epsilon)
            
            x_adv = x_adv_hat
            epsilon -= delta
            quert_count += 1
            
    return x_adv
