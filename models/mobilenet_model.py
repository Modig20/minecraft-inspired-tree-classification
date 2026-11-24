import torch
import torchvision
from torchvision.models import MobileNet_V3_Small_Weights
from torch import nn


def create_mobilenet(num_classes, pretrained=True):
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = torchvision.models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
