import torch
import torchvision
import torch.nn as nn
from torchvision.models import resnet18

model=resnet18(pretrained=False)
fc_features=model.fc.in_features
model.fc=nn.Linear(fc_features,1)
print(model)