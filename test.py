import torch
import torchinfo
from torchvision.models import alexnet

model = alexnet()

num_ftrs = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_ftrs, 8)

torchinfo.summary(model, (4, 3, 524, 524))

for i, data in enumerate(model.named_parameters()):
    name, param = data
    if "weight" in name:
        print(name)
        print(len(param.flatten()))


