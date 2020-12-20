# Get the pretrained model

import torch
import torchvision
from torch import nn as nn
from collections import OrderedDict


# obtain the pretrained resnet50
def Net(device):
    # net = torchvision.models.resnet18(pretrained=True)
    # net = torchvision.models.resnet34(pretrained=True)
    net = torchvision.models.resnet50(pretrained=True)
    # net = torchvision.models.vgg16(pretrained=True)
    # no fine tuning
    for param in net.parameters():
        param.requires_grad = False
    num_fc = net.fc.in_features
    net.fc = torch.nn.Linear(num_fc, 7)
    # num_fc = net.classifier[6].in_features
    # net.classifier[6] = torch.nn.Linear(num_fc, 7)
    net = net.to(device=device)
    return net

