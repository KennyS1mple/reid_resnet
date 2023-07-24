# CACIOUS CODING
# Data     : 7/24/23  5:37 PM
# File name: resnet
# Desc     :
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def resnet(reid_dim):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    set_parameter_requires_grad(model, False)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1024),
                             nn.BatchNorm1d(1024),
                             nn.ReLU(),
                             nn.Linear(1024, reid_dim),
                             nn.BatchNorm1d(reid_dim),
                             nn.ReLU())
    return model
