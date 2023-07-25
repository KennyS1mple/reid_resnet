# CACIOUS CODING
# Data     : 7/24/23  5:37 PM
# File name: resnet
# Desc     :
from torch import nn
from torchvision import models
from torchvision.models import ResNet34_Weights


def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def resnet(reid_dim, args):
    if args.use_pretrained:
        model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        set_parameter_requires_grad(model, False)
    else:
        model = models.resnet34()
    if args.use_relu:
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, reid_dim),
                                 nn.BatchNorm1d(reid_dim),
                                 nn.ReLU())
    else:
        model.fc = nn.Linear(model.fc.in_features, reid_dim)
    return model
