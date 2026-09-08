import torchvision
from torchvision.models import resnet18
import torch.nn as nn

model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=4)
print(model.fc)