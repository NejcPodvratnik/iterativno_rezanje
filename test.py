import numpy as np
import pandas as pd

import os
import cv2
import random
from tqdm import tqdm

from glob import glob
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

import torchinfo
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix

from dataset_chest_xray import *

# Path to save the checkpoints of the generated models
checkpoint_filepath = './'

chest_xray_dataset = ChestXRayDataset("./data/kaggle/kaggle/train3", "./data/train_age.csv")

train_size = int(0.7 * len(chest_xray_dataset))
val_size = int(0.15 * len(chest_xray_dataset))
test_size = len(chest_xray_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(chest_xray_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(1))

BATCH_SIZE = 64

train_loader = DataLoader(dataset = train_dataset, shuffle = True, batch_size = BATCH_SIZE)
val_loader = DataLoader(dataset = val_dataset, shuffle = False, batch_size = BATCH_SIZE)
test_loader = DataLoader(dataset = test_dataset, shuffle = False, batch_size = BATCH_SIZE)

class AgeModel(nn.Module):
    def __init__(self):
        super(AgeModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjust input size based on the feature map size after conv and pooling
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = AgeModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

torchinfo.summary(model, (BATCH_SIZE, 1, 64, 64))

# Define checkpoint directory
checkpoint_filepath = 'checkpoints/'
if not os.path.exists(checkpoint_filepath):
    os.makedirs(checkpoint_filepath)
age_checkpoint_path = checkpoint_filepath + 'age_classification.pth'

# Training loop
num_epochs = 20
best_val_loss = float('inf')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, total = len(train_loader.dataset)//train_loader.batch_size):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            val_loss += loss.item()
    
    running_loss /= len(train_loader)
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save the model if validation loss has decreased
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), age_checkpoint_path)
        print(f"Model saved at epoch {epoch+1}")

print("Training complete.")

# Load the best model from the checkpoint
model.load_state_dict(torch.load(age_checkpoint_path))

# Define evaluation function
def evaluate(model, test_loader):
    model.eval()
    mae = 0.0
    criterion = nn.L1Loss()  # MAE loss
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            mae += loss.item()
    mae /= len(test_loader)
    return mae

# Evaluate the model
mae = evaluate(model, test_loader)
print(f"Mean Absolute Error (MAE): {mae}")

# Generating the predictions from the test images
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        predictions.extend(outputs.squeeze().tolist())
        true_labels.extend(labels.tolist())

# Convert lists to numpy arrays for comparison
predictions = np.array(predictions)
true_labels = np.array(true_labels)

# Comparing the predict value with the correct value
selected_test_index = 2
print(f'Model predict: {predictions[selected_test_index]}')
print(f'Correct value: {true_labels[selected_test_index]}')
