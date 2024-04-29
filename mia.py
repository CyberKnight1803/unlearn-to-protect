import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics.classification import BinaryAccuracy
import numpy as np
from src.datamodule import CARCDataModule
from src.model import Model, VisionTransformer

class MIABinaryClassifier(L.LightningModule):
    def __init__(self):
        super(MIABinaryClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        outputs = self(data)
        loss = self.loss_fn(outputs.squeeze(), target)
        self.log("loss/train", loss)
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.01)

# Function to calculate loss for a given model and dataset
def calculate_loss(model, data_loader):
    model.eval()  # Ensure model is in evaluation mode
    loss_function = nn.CrossEntropyLoss()
    loss_values = []

    # with torch.no_grad():
    for imgs, labels in data_loader:
        outputs = model(imgs)
        loss = loss_function(outputs, labels)
        loss_values.append(loss.item())  # Store the scalar value of the loss

    return torch.tensor(loss_values, dtype=torch.float32)

# Load the models M_T, M_r, and M_f
feature_extractor_path_M_T = "./checkpoints/feature_extractor/resnet-18-r18-b64.pt"
classifier_path_M_T = "./checkpoints/classifier/resnet-18-r18-b64.pt"

feature_extractor_path_M_r = "./checkpoints/feature_extractor/resnet-18-r18-retain.pt"
classifier_path_M_r = "./checkpoints/classifier/resnet-18-r18-retain.pt"

# change this later
feature_extractor_path_M_f = "./checkpoints/feature_extractor/resnet-18-r18-forget.pt"
classifier_path_M_f = "./checkpoints/classifier/resnet-18-r18-forget.pt"

# Models
model_M_T = Model(model_name='resnet-18')
# model_M_T.feature_extractor.load_state_dict(torch.load(feature_extractor_path_M_T))
# model_M_T.classifier.load_state_dict(torch.load(classifier_path_M_T))

model_M_f = Model(model_name='resnet-18')
# model_M_f.feature_extractor.load_state_dict(torch.load(feature_extractor_path_M_f))
# model_M_f.classifier.load_state_dict(torch.load(classifier_path_M_f))

model_M_T.feature_extractor = torch.load(feature_extractor_path_M_T)
model_M_T.classifier = torch.load(classifier_path_M_T)

model_M_f.feature_extractor = torch.load(feature_extractor_path_M_f)
model_M_f.classifier = torch.load(classifier_path_M_f)

dm = CARCDataModule()

dm.setup()

# Calculate loss for D_train and D_test
train_loader = dm.train_dataloader()  # D_train
test_loader = dm.test_dataloader()  # D_test

train_loss_values = calculate_loss(model_M_T, train_loader)
test_loss_values = calculate_loss(model_M_f, test_loader)

# Create datasets for MIA
train_dataset = TensorDataset(train_loss_values, torch.ones(len(train_loss_values)))
test_dataset = TensorDataset(test_loss_values, torch.zeros(len(test_loss_values)))

# Create the MIA binary classifier and DataLoader
mia_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
train_size = int(0.8 * len(mia_dataset))
val_size = len(mia_dataset) - train_size
train_mia, val_mia = random_split(mia_dataset, [train_size, val_size])

train_loader = DataLoader(train_mia, batch_size=32, shuffle=True)
val_loader = DataLoader(val_mia, batch_size=32, shuffle=True)

mia_classifier = MIABinaryClassifier()
optimizer_mia = mia_classifier.configure_optimizers()

# Train the MIA binary classifier
trainer = L.Trainer(max_epochs=10)
trainer.fit(mia_classifier, train_loader)

# Evaluate the MIA classifier's accuracy
mia_classifier.eval()
accuracy = BinaryAccuracy()

with torch.no_grad():
    for data, target in val_loader:
        outputs = mia_classifier(data)
        predictions = (outputs.squeeze() > 0.5).float()
        accuracy.update(predictions, target)

final_accuracy = accuracy.compute()
print("MIA Classifier Accuracy:", final_accuracy.item())