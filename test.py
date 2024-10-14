import torch
import torch.nn as nn
import torchinfo
import torch.nn.functional as F
import numpy as np

from torchvision.models import resnet18
from torchvision.models import alexnet
from torchvision.models import vgg11


model = vgg11(weights=None)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_ftrs, 1)
torchinfo.summary(model, (64, 3, 256, 256))
#print(model)

sums = []
for name, param in model.named_parameters():
    if 'weight' in name:
        print(name)
        print(np.prod(np.shape(param.data)))

model = alexnet(weights=None)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_ftrs, 1)
torchinfo.summary(model, (64, 3, 256, 256))
#print(model)

sums = []
for name, param in model.named_parameters():
    if 'weight' in name:
        print(name)
        print(np.prod(np.shape(param.data)))

model = resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1)
torchinfo.summary(model, (64, 3, 256, 256))

sums = []
for name, param in model.named_parameters():
    if 'weight' in name:
        print(name)
        print(np.prod(np.shape(param.data)))
