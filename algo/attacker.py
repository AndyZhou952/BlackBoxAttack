from tqdm import tqdm 
import torch
import torch.nn.functional as F
from torch import nn

def NES_label_only_old(model, target_class, image, search_var, sample_num, g, u, mu, k):
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


def NES_label_only(model, image, target_class, search_var, m, mu, k):
    _, C, H, W = image.size()
    device = image.device

    # u: (n, C, H, W)
    u = torch.randn((m, C, H, W), device=device)

    # concat_u: [2n, C, H, W]
    concat_u = torch.cat([u, -u], dim=0)

    with torch.no_grad():
        perturbed_images = image.unsqueeze(0) + concat_u * search_var
        S_x_values = torch.zeros(2 * m, device=device)

        for i, perturbed_image in enumerate(perturbed_images):
            S_x_values[i] = S_x(model, perturbed_image, target_class, mu, m, k)

        # prob * concat_u: (2n,1, 1, 1) * (2n, C, H, W) = (2n, C, H, W)
        g = torch.sum(S_x_values.view(-1, 1, 1, 1) * concat_u, dim=0) / (2 * m * search_var)

    return g

#Query limited attack using NES
def NES(model, target_class, image, search_var, sample_num):
    #NES estimation
    model.eval()
    
    _, C,H,W = image.size()
    # u: (n, C, H, W)
    u = torch.randn((sample_num, C,H,W)).to(image.device)
    
    # cocnat_u: [2n, C, H, W]
    concat_u =  torch.cat([u , -u], dim=0)
    with torch.no_grad():
        # model output of dimension (2n, K)
        # prob of dimension (2n, 1)
        prob = F.softmax(model(image + concat_u * search_var), dim =1)[:,target_class].view(-1, 1, 1, 1)
        # g of dimension (C, H, W)
        # prob * concat_u: (2n,1, 1, 1) * (2n, C, H, W) = (2n, C, H, W)
        g = torch.sum(prob * concat_u, dim=0) / (2 * sample_num * search_var)
    return g

# image of dimension (batch_size, C, H, W)
def adversarial_generator(model, target_class, image, search_var, sample_num, bound, lr, query_limit):
    model.eval()
    
    device = next(model.parameters()).device
    image = image.to(device)
    batch_size, C, H, W = image.size()
    query_count = torch.zeros(batch_size).to(device)
    
    adv_images = list()
    with torch.no_grad():
        for i in range(batch_size):
            
            query = 0
            cur_img = image[i, :, :, :].unsqueeze(0)
            cur_target_class = target_class[i]
            adv_img = cur_img.clone()
            for _ in range(query_limit // (2 * sample_num)):
                g = NES(model, cur_target_class, adv_img, search_var, sample_num)
                adv_img = adv_img - lr * torch.sign(g)
                adv_img = torch.clamp(adv_img, min=cur_img-bound, max=cur_img + bound)
                adv_img = torch.clamp(adv_img, 0, 1)

                query += 2 * sample_num
                adv_logits = model(adv_img)
                adv_class = torch.argmax(adv_logits, dim=1)

                if adv_class != cur_target_class:
                    break
            query_count[i] = query
            adv_images.append(adv_img)
    return torch.concat(adv_images, dim=0), query_count


def get_rank_of_target(model, image, target_class, k = 5):
    """
    return the rank of the target in the top-k predictions
    return 0 if the target class is not in the top-k
    
    Expects 4D input
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

def S_x(model, image, target_class, mu, m, k):
    device = image.device
    R_sum = 0.0
    
    for _ in range(m):
        delta = (torch.rand_like(image) * 2 - 1) * mu
        perturbed_image = image + delta
        
        rank = get_rank_of_target(model, perturbed_image, target_class, k)
        R_sum += k - rank if rank != 0 else 0
    
    return R_sum / m  

def PIA_adversarial_generator(model, initial_images, reverse_mapping, adv_image_set, epsilon, 
                              delta, search_var, sample_num, eta_max, eta_min, bound, k, 
                              query_limit=30000, label_only=False, mu=None, m=None):
    """
    mu: label-only parameter
    """
    model.eval()
    
    device = next(model.parameters()).device
    initial_images = initial_images.to(device)
    target_images = list()
    target_classes = list()
    for initial_image in initial_images:
        # get the class with the second highest probability as the target class
        target_class = torch.topk(F.softmax(model(initial_image.unsqueeze(0)), dim = 1), k=5)[1][0][1].cpu().item()
        reversed_label = reverse_mapping[target_class]
        target_image = adv_image_set[reversed_label].to(device)
        target_images.append(target_image)
        target_classes.append(target_class)
    
    batch_size = initial_images.size(0)
    query_counts = torch.zeros(batch_size).to(device)
    adv_images = []

    with torch.no_grad():
        for i in range(batch_size):
            x = initial_images[i].unsqueeze(0)
            x_adv = target_images[i].unsqueeze(0)
            x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)

            query_count = 0
            # stopping criteria: target epsilon within the original image & the classification is the target class
            while epsilon > bound or not get_rank_of_target(model, x_adv, target_classes[i], k=1):
                if query_count >= query_limit:
                    break  # Non-convergence for this image

                if label_only:
                    gradient = NES_label_only(model, x_adv, target_classes[i], search_var, m, mu, k)
                else:
                    gradient = NES(model, target_classes[i], x_adv, search_var, sample_num)

                eta = eta_max
                x_adv_hat = x_adv - eta * gradient
                while not get_rank_of_target(model, x_adv_hat, target_classes[i], k):
                    if eta < eta_min:
                        epsilon += delta
                        delta /= 2
                        x_adv_hat = x_adv
                        break
                    eta /= 2
                    x_adv_hat = torch.clamp(x_adv - eta * gradient, x - epsilon, x + epsilon)
                
                x_adv = x_adv_hat
                epsilon -= delta
                query_count += 2 * sample_num

            query_counts[i] = query_count
            adv_images.append(x_adv)

    return torch.concat(adv_images, dim = 0), query_counts