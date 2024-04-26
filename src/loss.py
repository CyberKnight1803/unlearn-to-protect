from typing import Union
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from src.config import Config

class ContrastiveUnlearnLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveUnlearnLoss, self).__init__()
        self.tau = temperature
        self.eps = 1e-9
    
    def forward(self, h_f, labels_f, h_r, labels_r):
        # Assuming imgs_f and imgs_r are the embeddings z_i and z_r
        batch_size = h_f.size(0)

        # Compute cosine similarities between embeddings
        sim_matrix = F.cosine_similarity(
            h_f.unsqueeze(1), 
            h_r.unsqueeze(0), 
            dim=2
        )

        # Create masks for positive and negative samples
        labels_f = labels_f.view(-1, 1)
        labels_r = labels_r.view(1, -1)
        p_msk = torch.eq(labels_f, labels_r)
        n_msk = torch.ne(labels_f, labels_r)

        # Compute the loss
        loss = 0
        for i in range(batch_size):
            p_idx = p_msk[i]
            n_idx = n_msk[i]

            p_sim = sim_matrix[i][p_idx]
            n_sim = sim_matrix[i][n_idx]

            # Loss
            loss = -1 * torch.sum(
                torch.log(
                    torch.exp(n_sim) / torch.sum(torch.exp(p_sim)) + self.eps 
                )
            ) / (n_sim.size(0) + 1)

        return loss / batch_size