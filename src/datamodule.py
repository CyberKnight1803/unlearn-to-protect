from typing import Optional
import os 
import numpy as np
import pandas as pd 
from PIL import Image 
import torch 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms 
import lightning as L 
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from src.config import Config

class CARCDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        data_file: str, 
        transform,
        device: str,
    ):
        self.image_dir = image_dir
        self.data = pd.read_csv(data_file)
        self.transform = transform
        self.device = device 

    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, idx):
        name = self.data.iloc[idx]['name']
        image_path = self.data.iloc[idx]['file']
        label = self.data.iloc[idx]['age_bin']
        
        img = Image.open(os.path.join(self.image_dir, image_path))
        img = self.transform(img)
        
        return name, img.to(device=self.device), torch.tensor(label, device=self.device)

class CARCDataModule(L.LightningDataModule):
    def __init__(
        self, 
        image_dir: str = Config.PATH_DATASET,
        data_files: dict[str] = Config.DATA_FILES,
        retain: bool = False,
        batch_size: int = Config.BATCH_SIZE, 
        device: str = Config.ACCELERATOR
    ) -> None: 
        
        super(CARCDataModule, self).__init__()
        self.save_hyperparameters(logger=True)
        
        self.image_dir = image_dir
        self.data_files = data_files 
        self.batch_size = batch_size 
        
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Grayscale()
        ])
        
        self.retain = retain
        self.device = device
        
    def setup(self, stage: Optional[str] = None) -> None:
        
        if self.retain:
            self.train_dataset =  CARCDataset(
                image_dir=self.image_dir,
                data_file=self.data_files['retain'],
                transform=self.transform,
                device=self.device
            ) 
        else: 
            self.train_dataset =  CARCDataset(
                image_dir=self.image_dir,
                data_file=self.data_files['train'],
                transform=self.transform,
                device=self.device
            ) 
        
        self.test_dataset = CARCDataset(
            image_dir=self.image_dir,
            data_file=self.data_files['test'], 
            transform=self.transform,
            device=self.device
        )
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
        )


class ContrastivePairsDataset(Dataset):
    def __init__(self, retain_dataset, forget_dataset):
        self.retain_dataset = retain_dataset
        self.forget_dataset = forget_dataset 
        
    def __len__(self):
        return len(self.forget_dataset)
    
    def __getitem__(self, idx):
        names_f, imgs_f, labels_f = self.forget_dataset[idx]
        
        r_idx = np.random.randint(0, len(self.retain_dataset))
        names_r, imgs_r, labels_r = self.retain_dataset[r_idx]

        return names_f, imgs_f, labels_f, names_r, imgs_r, labels_r
class ContrastiveDataModule(L.LightningDataModule):
    def __init__(
        self, 
        image_dir: str = Config.PATH_DATASET,
        data_files: str = Config.RETAIN_DATA_FILES,
        batch_size: int = Config.BATCH_SIZE,
        device: str = Config.ACCELERATOR
    ) -> None:
        
        super(ContrastiveDataModule, self).__init__()
        self.save_hyperparameters(logger=True)
        
        self.image_dir = image_dir
        self.data_files = data_files
        self.batch_size = batch_size
        self.device = device
        
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Grayscale()
        ])
         
    
    def setup(self, stage: Optional[str] = None) -> None:
        self.retain_dataset = CARCDataset(
            image_dir=self.image_dir,
            data_file=self.data_files['retain'],
            transform=self.transform,
            device=self.device
        )
            
        self.forget_dataset = CARCDataset(
            image_dir=self.image_dir,
            data_file=self.data_files['forget'],
            transform=self.transform,
            device=self.device
        )
        
        self.train_dataset = ContrastivePairsDataset(
            retain_dataset=self.retain_dataset,
            forget_dataset=self.forget_dataset, 
        )
        
        # Pairs are formed randomly => train pairs != test pairs
        self.test_dataset = ContrastivePairsDataset(
            retain_dataset=self.retain_dataset,
            forget_dataset=self.forget_dataset, 
        )
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size, 
            shuffle=True
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
        )