import torch
import numpy as np
import torch.nn.functional as F

# https://github.com/facebookresearch/swav/blob/main/main_swav.py

def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


class SinkhornKnopp_imb(torch.nn.Module):
        def __init__(self, args, cls_dist):
            super().__init__()
            self.num_iters = args.num_iters_sk
            self.epsilon = args.epsilon_sk
            self.temperature = args.cos_temp
            self.cls_dist = cls_dist 
            
        @torch.no_grad()
        def iterate(self, Q):
            
            Q = shoot_infs(Q)
            sum_Q = torch.sum(Q)
            Q /= sum_Q
            
            B = Q.shape[1]
            K = Q.shape[0]
             
            for it in range(self.num_iters):
                
                sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
                Q /= sum_of_rows
                Q = shoot_infs(Q)
                Q *= self.cls_dist
                
                # normalize each column: total weight per sample must be 1/B
                Q /= torch.sum(Q, dim=0, keepdim=True)
                Q /= B

            Q *= B # the colomns must sum to 1 so that Q is an assignment
                    
            return Q.t()

        @torch.no_grad()
        def forward(self, embeddings, centroids):
            
            
            last_token_rep = F.normalize(embeddings, p=2, dim=-1)
            centroids = F.normalize(centroids, p=2, dim=-1)
            
            # Compute cosine similarity (which is equivalent to the dot product for normalized vectors)
            similarities = torch.matmul(last_token_rep, centroids.T)  

            # Apply the temperature scaling factor (similar to dividing by τ in the equation)
            similarities = similarities / self.temperature
        
            # Convert similarities to probability distributions using softmax
            pt = F.softmax(similarities, dim=-1) 
            
            # Compute the OT loss as the cross-entropy between pseudo-labels and pt
            pt = torch.log(pt + 1e-8)
            
            # Divide by temperature (epsilon) to scale the distance
            q =  pt / (self.epsilon)
            
            # Apply exponential to form soft assignment weights
            q = torch.exp(q).t()
            
            return self.iterate(q)