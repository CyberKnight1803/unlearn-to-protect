from typing import List, Dict 
import torch
import torch.nn as nn 
import torch.nn.functional as F 

from torchvision import models

import lightning as L 
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics.functional import accuracy, f1_score
import transformers as T

from src.loss import ContrastiveUnlearnLoss
from src.config import Config


class FeatureExtractor(nn.Module):
    accepted_models = [
        'resnet-18', 
        'resnet-34', 
        'resnet-50', 
        'resnet-101',
        'resnet-152', 
    ]
    
    def __init__(
        self,
        model_name: str = Config.MODEL_NAME, 
        feature_size: int = Config.FEATURE_SIZE
    ) -> None: 
        super(FeatureExtractor, self).__init__()
        
        # Check if model name is correct
        assert model_name in self.accepted_models, f"Model: {model_name} not in {self.accepted_models}"
        
        if model_name == 'resnet-18':
            self.model = models.resnet18(num_classes=feature_size, pretrained=False)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), padding=(3, 3), bias=False)
        
        elif model_name == 'resnet-34':
            self.model = models.resnet34(num_classes=feature_size, pretrained=False)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), padding=(3, 3), bias=False)
        
        elif model_name == 'resnet-50':
            self.model = models.resnet50(num_classes=feature_size, pretrained=False)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), padding=(3, 3), bias=False)
        
        elif model_name == 'resnet-101':
            self.model = models.resnet50(num_classes=feature_size, pretrained=False)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), padding=(3, 3), bias=False)
        
        elif model_name == 'resnet-152':
            self.model = models.resnet152(num_classes=feature_size, pretrained=False)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), padding=(3, 3), bias=False)

    def forward(self, x):
        return self.model(x)

class MLPLayer(nn.Module):
    def __init__(
        self, 
        feature_size: int = Config.FEATURE_SIZE,
        num_classes: int = Config.NUM_CLASSES
    ) -> None:
        
        super(MLPLayer, self).__init__()
        
        self.mlp_layer = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.LayerNorm(256), 
            nn.ReLU(), 
            nn.Linear(256, 64), 
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, num_classes), 
            nn.LayerNorm(num_classes)
        )
    
    def forward(self, x):
        return self.mlp_layer(x)

class Model(L.LightningModule):
    def __init__(
        self, 
        model_name: Config.MODEL_NAME, 
        feature_size: int = Config.FEATURE_SIZE, 
        num_classes: int = Config.NUM_CLASSES,
        learning_rate: float = Config.LEARNING_RATE,
        weight_decay: float = Config.WEIGHT_DECAY
    ) -> None:
        
        super(Model, self).__init__()
        self.save_hyperparameters(logger=True)
        
        self.feature_extractor = FeatureExtractor(model_name, feature_size)
        self.classifier = MLPLayer(feature_size, num_classes)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extractor(x)
        out = self.classifier(out)
        return out
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        names, imgs, labels = batch 
        out = self(imgs)
        
        loss = self.loss_fn(out, labels)
        acc = accuracy(out, labels, num_classes=self.hparams.num_classes, task='multiclass')
        f1 = f1_score(out, labels, num_classes=self.hparams.num_classes, task='multiclass', average='macro')
        
        self.log_dict({
            'loss/train': loss,
            'acc/train': acc,
            'f1/train': f1
        })
        
        return {
            'loss': loss, 
            'acc': acc,
            'f1': f1
        }
         
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        names, imgs, labels = batch 
        out = self(imgs)
        
        loss = self.loss_fn(out, labels)
        acc = accuracy(out, labels, num_classes=self.hparams.num_classes, task='multiclass')
        f1 = f1_score(out, labels, num_classes=self.hparams.num_classes, task='multiclass', average='macro')
        
        self.log_dict({
            'loss/val': loss,
            'acc/val': acc,
            'f1/val': f1
        })
        
        return {
            'loss': loss, 
            'acc': acc,
            'f1': f1
        }
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        names, imgs, labels = batch 
        out = self(imgs)
        
        loss = self.loss_fn(out, labels)
        acc = accuracy(out, labels, num_classes=self.hparams.num_classes, task='multiclass')
        f1 = f1_score(out, labels, num_classes=self.hparams.num_classes, task='multiclass', average='macro')
        
        self.log_dict({
            'loss/test': loss,
            'acc/test': acc,
            'f1/test': f1
        })
        
        return {
            'loss': loss, 
            'acc': acc,
            'f1': f1
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay    
        )
    
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=1000
        )
        
        return [optimizer], [lr_scheduler]
    
class VisionTransformer(L.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_classes: int = 8
    ) -> None:
        super(VisionTransformer, self).__init__()
        self.save_hyperparameters(logger=True)
        
        self.config = T.ViTConfig(image_size=128)
        self.config.num_labels = num_classes        
        self.model = T.ViTForImageClassification(config=self.config)
    
    def forward(self, x, y):
        return self.model(x, labels=y)
    
    def training_step(self, batch, batch_idx):
        imgs, labels = batch 
        out = self(imgs, labels)
        
        loss = out.loss
        acc = accuracy(out.logits, labels, num_classes=self.hparams.num_classes, task='multiclass')
        f1 = f1_score(out.logits, labels, num_classes=self.hparams.num_classes, task='multiclass', average='macro')
        
        self.log_dict({
            'loss/train': loss,
            'acc/train': acc,
            'f1/train': f1
        })
        
        return {
            'loss': loss, 
            'acc': acc,
            'f1': f1
        }
        
        
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch 
        out = self(imgs, labels)
        
        loss = out.loss
        acc = accuracy(out.logits, labels, num_classes=self.hparams.num_classes, task='multiclass')
        f1 = f1_score(out.logits, labels, num_classes=self.hparams.num_classes, task='multiclass', average='macro')
        
        self.log_dict({
            'loss/val': loss,
            'acc/val': acc,
            'f1/val': f1
        })
        
        return {
            'loss': loss, 
            'acc': acc,
            'f1': f1
        } 
    
    def test_step(self, batch, batch_idx):
        imgs, labels = batch 
        out = self(imgs, labels)
        
        loss = out.loss
        acc = accuracy(out.logits, labels, num_classes=self.hparams.num_classes, task='multiclass')
        f1 = f1_score(out.logits, labels, num_classes=self.hparams.num_classes, task='multiclass', average='macro')
        
        self.log_dict({
            'loss/test': loss,
            'acc/test': acc,
            'f1/test': f1
        })
        
        return {
            'loss': loss, 
            'acc': acc,
            'f1': f1
        }
        
    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        
    
    
class ContrastiveLearning(L.LightningModule):
    def __init__(
        self,
        feature_extractor_path: str = Config.DEFAULT_FEATURE_EXTRACTOR_WEIGHTS,
        classifier_path: str = Config.DEFAULT_CLASSIFIER_WEIGHTS,
        model_name: str = Config.MODEL_NAME, 
        feature_size: int = Config.FEATURE_SIZE,
        num_classes: int = Config.NUM_CLASSES,
        learning_rate: float = Config.LEARNING_RATE,
        weight_decay: float = Config.WEIGHT_DECAY, 
        temperature: float = Config.TEMPERATURE
    ) -> None:
        
        super(ContrastiveLearning, self).__init__()
        self.save_hyperparameters(logger=True)
        
        self.feature_extractor = torch.load(feature_extractor_path)
        self.classifier = torch.load(classifier_path)
        
        self.contrastive_loss_fn = ContrastiveUnlearnLoss(temperature=temperature)
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x_f, x_r) -> List[torch.Tensor]:
        h_f = self.feature_extractor(x_f)
        h_r = self.feature_extractor(x_r)
        out_r = self.classifier(h_r)
        return h_f, h_r, out_r
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # forget, retain
        names_f, imgs_f, labels_f, names_r, imgs_r, labels_r = batch
        
        # Forward 
        h_f, h_r, out_r = self(imgs_f, imgs_r)   
        loss = 0.6 * self.contrastive_loss_fn(h_f, labels_f, h_r, labels_r) + 0.4 * self.cross_entropy_loss_fn(out_r, labels_r)
        acc = accuracy(out_r, labels_r, num_classes=self.hparams.num_classes, task='multiclass')
        f1 = f1_score(out_r, labels_r, num_classes=self.hparams.num_classes, task='multiclass', average='macro')
        
        self.log_dict({
            'loss/train': loss,
            'acc/train': acc,
            'f1/train': f1
        })
        
        return {
            'loss': loss, 
            'acc': acc,
            'f1': f1
        } 
        
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # forget, retain
        names_f, imgs_f, labels_f, names_r, imgs_r, labels_r = batch
        
        # Forward 
        h_f, h_r, out_r = self(imgs_f, imgs_r)   
        loss = 0.6 * self.contrastive_loss_fn(h_f, labels_f, h_r, labels_r) + 0.4 * self.cross_entropy_loss_fn(out_r, labels_r)
        acc = accuracy(out_r, labels_r, num_classes=self.hparams.num_classes, task='multiclass')
        f1 = f1_score(out_r, labels_r, num_classes=self.hparams.num_classes, task='multiclass', average='macro')
        
        self.log_dict({
            'loss/val': loss,
            'acc/val': acc,
            'f1/val': f1
        })
        
        return {
            'loss': loss, 
            'acc': acc,
            'f1': f1
        } 
        
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # forget, retain
        names_f, imgs_f, labels_f, names_r, imgs_r, labels_r = batch
        
        # Forward 
        h_f, h_r, out_r = self(imgs_f, imgs_r)   
        loss = 0.6 * self.contrastive_loss_fn(h_f, labels_f, h_r, labels_r) + 0.4 * self.cross_entropy_loss_fn(out_r, labels_r)
        acc = accuracy(out_r, labels_r, num_classes=self.hparams.num_classes, task='multiclass')
        f1 = f1_score(out_r, labels_r, num_classes=self.hparams.num_classes, task='multiclass', average='macro')
        
        self.log_dict({
            'loss/test': loss,
            'acc/test': acc,
            'f1/test': f1
        })
        
        return {
            'loss': loss, 
            'acc': acc,
            'f1': f1
        } 
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay    
        )
    
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=1000
        )
        
        return [optimizer], [lr_scheduler]
    
        
        