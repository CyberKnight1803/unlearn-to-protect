import torch
import torch.nn as nn 
import torch.nn.functional as F 

import lightning as L 
from lightning.pytorch.utilities.types import STEP_OUTPUT

class Evaluator(L.LightningModule):
    def __init__(
        self, 
        feature_size: int = 1024
    ) -> None: 
        super(Evaluator, self).__init__()
        self.save_hyperparameters(logger=True)
        
        self.net = nn.Sequential(
            nn.Linear(feature_size, 512), 
            nn.ReLU(), 
            nn.Linear(512, 128), 
            nn.ReLU(),
            nn.Linear(128, 32), 
            nn.ReLU(),
            nn.Linear(32, 8), 
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x):
        return self.net(x)

    def training_step(self, *args: torch.Any, **kwargs: torch.Any) -> STEP_OUTPUT:
        return super().training_step(*args, **kwargs)

    def validation_step(self, *args: torch.Any, **kwargs: torch.Any) -> STEP_OUTPUT:
        return super().validation_step(*args, **kwargs)
    
    def configure_optimizers(self):
        return super().configure_optimizers()