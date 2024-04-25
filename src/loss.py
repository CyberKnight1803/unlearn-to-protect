import torch 
import torch.nn as nn 


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float
    ) -> None:
        super(ContrastiveLoss, self).__init__()
        
        self.temperature = temperature
    
    def forward(self):
        pass 
