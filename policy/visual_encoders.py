from .backbones.resnet import *
from .backbones.mobilenet import *
from torch import nn as nn

from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
}


def group_norm_wrapper(ngroups):
    def group_norm(out_channels):
        if out_channels % ngroups == 0:
            return nn.GroupNorm(ngroups, out_channels)
        else:
            return nn.Identity()

    return group_norm

def mobilenet_v2(in_channels, base_planes, ngroups, pretrained: bool = False, progress: bool = True) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(norm_layer=group_norm_wrapper(ngroups))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["mobilenet_v2"], progress=progress)
        model.load_state_dict(state_dict)

    print("Visual encoder total parameters: ", sum(p.numel() for p in model.parameters()))
    return model


def resnet18(in_channels, base_planes, ngroups):
    model = ResNet(in_channels, base_planes, ngroups, BasicBlock, [2, 2, 2, 2])

    return model


def resnet50(in_channels: int, base_planes: int, ngroups: int) -> ResNet:
    model = ResNet(in_channels, base_planes, ngroups, Bottleneck, [3, 4, 6, 3])

    print("Visual encoder total parameters: ", sum(p.numel() for p in model.parameters()))
    return model


def resneXt50(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        ResNeXtBottleneck,
        [3, 4, 6, 3],
        cardinality=int(base_planes / 2),
    )

    return model


def se_resnet50(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels, base_planes, ngroups, SEBottleneck, [3, 4, 6, 3]
    )

    return model


def se_resneXt50(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        SEResNeXtBottleneck,
        [3, 4, 6, 3],
        cardinality=int(base_planes / 2),
    )

    return model


def se_resneXt101(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        SEResNeXtBottleneck,
        [3, 4, 23, 3],
        cardinality=int(base_planes / 2),
    )

    return model
