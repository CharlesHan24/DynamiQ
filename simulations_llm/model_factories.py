import models
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import pdb

# def ResNet9(num_classes, color_channels): return models.ResNet9(num_classes, color_channels)
# def ResNet18(num_classes, color_channels): return models.ResNet18(num_classes, color_channels)
# def ResNet34(num_classes, color_channels): return models.ResNet34(num_classes, color_channels)
def ResNet34(num_classes, color_channels): return torchvision.models.resnet34(num_classes=num_classes)
def ResNet18(num_classes, color_channels): return torchvision.models.resnet18(num_classes=num_classes)
def ResNet50(num_classes, color_channels): return torchvision.models.resnet50(num_classes=num_classes)
def ResNet101(num_classes, color_channels): return torchvision.models.resnet101(num_classes=num_classes)
def shufflenet(num_classes, color_channels):
    return torchvision.models.shufflenet_v2_x1_5(num_classes=num_classes)

def squeezenet(num_classes, color_channels):
    return torchvision.models.squeezenet1_1(num_classes=num_classes, dropout=0)


def get_model_size(model: nn.Module):
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * 4
        if param.dtype != torch.float32:
            pdb.set_trace()
    return total_size

def VGG19(num_classes, color_channels):
    model = torchvision.models.vgg19(num_classes=num_classes)
    print("Using VGG19, size: {}".format(get_model_size(model)))
    return model
    
def VGG16(num_classes, color_channels):
    model = torchvision.models.vgg16(num_classes=num_classes)
    print("Using VGG19, size: {}".format(get_model_size(model)))
    return model


class Net(nn.Module):
    def __init__(self, num_classes, color_channels):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(color_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)