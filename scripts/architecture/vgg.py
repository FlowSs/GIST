# Adapted from CIFAR-ZOO
# -*-coding:utf-8-*-
import torch.nn as nn

__all__ = ["vgg11", "vgg13", "vgg16", "vgg19"]

cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, input_channel=3, feat_ext=None):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self.in_channels = input_channel
        self._initialize_weights()

    def forward(self, x, hook=False):
        x = self.features(x)
        x_flat = x.view(x.size(0), -1)
        x = self.classifier(x_flat)
        
        if hook:
            return {'pred': x.detach().cpu().numpy(), 'x_flat': x_flat.detach().cpu().numpy()}
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg11(num_classes, input_channel=3):
    return VGG(make_layers(cfg["A"], input_channel, batch_norm=True), num_classes)


def vgg13(num_classes, input_channel=3):
    return VGG(make_layers(cfg["B"], input_channel, batch_norm=True), num_classes)


def vgg16(num_classes, input_channel=3):
    return VGG(make_layers(cfg["D"], input_channel, batch_norm=True), num_classes, feat_ext=40)


def vgg19(num_classes, input_channel=3):
    return VGG(make_layers(cfg["E"], input_channel, batch_norm=True), num_classes, feat_ext=49)
