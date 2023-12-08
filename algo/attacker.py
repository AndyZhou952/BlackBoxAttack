from tqdm import tqdm 
import torch
import torch.nn.functional as F
from torch import nn

def mask_top_k(output, k):
    # output: (batch_size, K)
    threshold = torch.topk(output, k, dim=1).values[:, k-1]
    # threshold: (batch_size, 100)
    threshold = threshold.unsqueeze(1).expand_as(output)
    output[output < threshold] = float('nan')
    return output 

def NES_label_only(model, image, target_class, search_var,sample_num, m, mu, k):
    model.eval()
    
    # image: (1, C, H, W)
    _, C, H, W = image.size()
    device = image.device

    # u: (n, C, H, W)
    u = torch.randn((sample_num, C, H, W), device=device)

    # concat_u: [2n, C, H, W]
    concat_u = torch.cat([u, -u], dim=0)

    with torch.no_grad():
        # (2n, C, H, W) = (1, C, H, W) + (2n, C, H, W)
        perturbed_images = image + concat_u * search_var
        
        # (2n, )
        proxy_prob = torch.zeros(2 * sample_num, device=device)

        for i, perturbed_image in enumerate(perturbed_images):
            # target_class: (0) scaler tensor
            proxy_prob[i] = S_x(model, perturbed_image.unsqueeze(0), target_class, mu, m, k)

        # prob * concat_u: (2n,1, 1, 1) * (2n, C, H, W) = (2n, C, H, W)
        g = torch.sum(proxy_prob.view(-1, 1, 1, 1) * concat_u, dim=0) / (2 * sample_num * search_var)

    return g

# def NES_label_only(model, image, target_class, search_var,sample_num, m, mu, k):
#     model.eval()
    
#     # image: (1, C, H, W)
#     _, C, H, W = image.size()
#     device = image.device

#     # u: (n, C, H, W)
#     u = torch.randn((sample_num, C, H, W), device=device)

#     # concat_u: [2n, C, H, W]
#     concat_u = torch.cat([u, -u], dim=0)

#     with torch.no_grad():
#         # (2n, C, H, W) = (1, C, H, W) + (2n, C, H, W)
#         perturbed_images = image + concat_u * search_var

#         # target_class: (0) scaler tensor
#         # proxy_prob: (2n, )
#         proxy_prob = S_x(model, perturbed_images, target_class, mu, m, k)
        
#         print(proxy_prob.shape)
        
#         # prob * concat_u: (2n,1, 1, 1) * (2n, C, H, W) = (2n, C, H, W)
#         g = torch.sum(proxy_prob.view(-1, 1, 1, 1) * concat_u, dim=0) / (2 * sample_num * search_var)

#     return g

# def NES_label_only(model, image, target_class, search_var,sample_num, m, mu, k):
#     model.eval()
    
#     # image: (1, C, H, W)
#     _, C, H, W = image.size()
#     device = image.device

#     # u: (n, C, H, W)
#     u = torch.randn((sample_num, C, H, W), device=device)

#     with torch.no_grad():
#         # (n, C, H, W) = (1, C, H, W) + (n, C, H, W)
#         perturbed_images = image + u * search_var

#         # target_class: (0) scaler tensor
#         # proxy_prob: (n, )
#         proxy_prob = S_x(model, perturbed_images, target_class, mu, m, k)
        
#         # prob * concat_u: (n,1, 1, 1) * (n, C, H, W) = (n, C, H, W)
#         g_pos = torch.sum(proxy_prob.view(-1, 1, 1, 1) * u, dim=0) / (sample_num * search_var)

#         perturbed_images = image - u * search_var
#         proxy_prob = S_x(model, perturbed_images, target_class, mu, m, k)
#         g_neg = torch.sum(-proxy_prob.view(-1, 1, 1, 1) * u, dim=0) / (sample_num * search_var)
        
#     return g_pos/2 + g_neg /2


def NES(model, target_class, image, search_var, sample_num, k=None):
    #NES estimation
    model.eval()
    
    _, C,H,W = image.size()
    # u: (n, C, H, W)
    u = torch.randn((sample_num, C,H,W)).to(image.device)
    
    # cocnat_u: [2n, C, H, W]
    concat_u =  torch.cat([u , -u], dim=0)
    with torch.no_grad():
        # model output of dimension (2n, K)
        # prob of dimension (2n, 1, 1, 1)
        
        output = F.softmax(model(image + concat_u * search_var), dim =1)
        if k is not None:
            output = mask_top_k(output, k)
        prob = output[:,target_class].view(-1, 1, 1, 1)
        # g of dimension (C, H, W)
        # prob * concat_u: (2n,1, 1, 1) * (2n, C, H, W) = (2n, C, H, W)
        g = torch.nanmean(prob * concat_u, dim=0) / search_var
        g = torch.nan_to_num(g, nan=0.0)
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
                # g = NES(model, cur_target_class, adv_img, search_var, sample_num)
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
    # image: (batch_size, C, H, W)
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        # output: (batch_size, K) <-- Big K: number of classes
        probabilities = F.softmax(output, dim = 1)
        
        # top_classes: (batch_size, k) <-- small k: top k
        top_probs, top_classes = torch.topk(probabilities, k)
        # target_class: (0) scalar tensor
        
        # matches: (batch_size, k)
        matches = (target_class == top_classes).to(int)
        # indices: (batch_size, 1)
        ranks = torch.argmax(matches, dim=1)
        ranks = ranks + 1
        ranks[~matches.any(dim=1)] = 0
        
        # ranks: (batch_size)
        return ranks
    
# def get_rank_of_target(model, image, target_class, k = 5):
#     """
#     return the rank of the target in the top-k predictions
#     return 0 if the target class is not in the top-k
    
#     Expects 4D input
#     """
    
#     device = next(model.parameters()).device
#     # image: (batch_size, C, H, W)
#     image = image.to(device)
    
#     with torch.no_grad():
#         output = model(image)
#         # output: (batch_size, K) <-- Big K: number of classes
#         output = torch.nan_to_num(output, nan=0.0)
#         probabilities = F.softmax(output, dim = 1)
#         # top_classes: (batch_size, k) <-- small k: top k
#         top_probs, top_classes = torch.topk(probabilities, k)
#         # target_class: (0) scalar tensor
        
        
#         if target_class in top_classes[0]:
#             target_rank = (top_classes[0] == target_class).nonzero(as_tuple = True)[0].item() + 1
#             return target_rank
#         else:
#             return 0

def S_x(model, image, target_class, mu, m, k):
    
    # image: (n, C, H, W)
    device = image.device
    n, C, H, W = image.size()
    
    # image: (n, 1, C, H, W)
    image = image.unsqueeze(1)
    
    # delta: (m, C, H, W)
    delta = (torch.rand((m, C, H, W)) * 2 - 1) * mu
    
    # delta: (1, m, C, H, W)
    delta = delta.unsqueeze(0)
    
    # perturbed_image: (n * m, C, H, W)
    perturbed_image = (image + delta.to(device)).view(-1, C, H, W)

    # ranks: (n * m) 
    ranks = get_rank_of_target(model, perturbed_image, target_class, k)
    R = (5 - ranks) * (ranks > 0)
    
    # R: (n)
    R = R.view(n, m).to(float).mean(dim=1)
    
    return R

# def S_x(model, image, target_class, mu, m, k):
    
#     # image: (1, C, H, W)
#     device = image.device
#     _, C, H, W = image.size()
    
#     R_sum = 0.0
    
#     # delta: (m, C, H, W)
#     delta = (torch.rand((m, C, H, W)) * 2 - 1) * mu
#     perturbed_image = image + delta.to(device)

#     # ranks: (m) 
#     ranks = get_rank_of_target(model, perturbed_image, target_class, k)
#     R = (5 - ranks) * (ranks > 0)
#     # R_sum += k - rank if rank != 0 else 0
        
# #     for _ in range(m):
# #         # delta: (m, 
# #         delta = (torch.rand_like(image) * 2 - 1) * mu
# #         perturbed_image = image + delta
        
# #         rank = get_rank_of_target(model, perturbed_image, target_class, k)
# #         R_sum += k - rank if rank != 0 else 0
    
#     return R.to(float).mean() 

# 
def PIA_adversarial_generator(model, images, sample_img_dataset, epsilon, 
                              delta, search_var, sample_num, eta_max, eta_min, bound, k, 
                              query_limit=30000, label_only=False, mu=None, m=None):
    """
    mu: label-only parameter
    """
    model.eval()
    
    device = next(model.parameters()).device
    images = images.to(device)
    batch_size, C, H, W = images.size()
    
    # dimension: (batch_size, 100) -> (batch_size, )
    output = model(images)
    
    # target_classes = torch.topk(F.softmax(output, dim=1), dim= 1, largest=True, sorted=True, k=2).indices[:, 1]
    target_classes = torch.topk(F.softmax(output, dim=1), dim= 1, largest=True, sorted=True, k=5).indices[:, 1]
    
    query_counts = torch.zeros(batch_size)
    adv_images = list()

    with torch.no_grad():
        for i in range(batch_size):
            x = images[i, :, :, :].unsqueeze(0)
            # y_adv: (0) scaler tensor
            y_adv = target_classes[i]
            x_adv = sample_img_dataset[y_adv, :, :, :].unsqueeze(0).to(device)
            
            x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
            
            query_count = 0
            # stopping criteria: target epsilon within the original image & the classification is the target class
            
            while epsilon > bound or  (get_rank_of_target(model, x_adv, target_classes[i], k=1) != 1):
                if query_count >= query_limit:
                    break  # Non-convergence for this image

                if label_only:
                    gradient = NES_label_only(model, x_adv,y_adv, search_var,sample_num,  m, mu, k)
                else:
                    gradient = NES(model, y_adv, x_adv, search_var, sample_num, k=k)
                
                # exp
                gradient = torch.sign(gradient)
                
                eta = eta_max
                # x_adv_hat = x_adv + eta * gradient
                x_adv_hat = torch.clamp(x_adv + eta * gradient, x - epsilon, x + epsilon)
                x_adv_hat = torch.clamp(x_adv_hat, 0, 1)
                
                # while not get_rank_of_target(model, x_adv_hat, y_adv, k):
                while not get_rank_of_target(model, x_adv_hat, y_adv, 1):
                    if eta < eta_min:
                        epsilon += delta
                        delta /= 2
                        x_adv_hat = x_adv
                        break
                    eta /= 2
                    x_adv_hat = torch.clamp(x_adv + eta * gradient, x - epsilon, x + epsilon)
                    x_adv_hat = torch.clamp(x_adv_hat, 0, 1)
                    
                # x_adv_hat = torch.clamp(x_adv_hat, x - epsilon, x + epsilon)
                # x_adv_hat = torch.clamp(x_adv_hat, 0, 1)
                    
                x_adv = x_adv_hat
                epsilon -= delta
                
                if label_only:
                    query_count += 2 * sample_num * m
                else:
                    query_count += 2 * sample_num
                    
                if epsilon < 0:
                    break
                
            query_counts[i] = query_count
            adv_images.append(x_adv)

    return torch.concat(adv_images, dim = 0), query_counts




# def NES_partial_info(model, target_class, image, search_var, sample_num, k):
#     model.eval()
#     _, C, H, W = image.size()
#     device = image.device
#     # u: (n, C, H, W)
#     u = torch.randn((sample_num, C,H,W)).to(image.device)
    
#     # cocnat_u: [2n, C, H, W]
#     concat_u =  torch.cat([u , -u], dim=0)
#     perturbed_images = image + concat_u * search_var
#     perturbed_images = perturbed_images.view(-1, C, H, W)

#     gradients = torch.zeros_like(image)
#     with torch.no_grad():
#         outputs = model(perturbed_images)
#         top_probs, top_classes = torch.topk(F.softmax(outputs, dim=1), k)

#         # subset for the parts where the target class is in the top k classes
#         target_in_top_k = top_classes == target_class
#         # TODO: when not exist then 0?

#         g = torch.sum(target_probs.view(-1, 1, 1, 1) * concat_u, dim=0) / (2 * sample_num * search_var)

#     return g