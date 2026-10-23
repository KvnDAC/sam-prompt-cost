"""Small RGB controls: declared variants, not reproductions of published recipes."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .unet import UNet


def norm(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)


class Conv(nn.Sequential):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False),
                         norm(outgoing), nn.ReLU(inplace=True))


class Separable(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, incoming, 3, padding=1, groups=incoming, bias=False),
                         norm(incoming), nn.ReLU(inplace=True),
                         nn.Conv2d(incoming, outgoing, 1, bias=False), norm(outgoing), nn.ReLU(inplace=True))


class DSUNet(nn.Module):
    """Three-level U-shaped depthwise-separable network, bilinear upsampling."""
    def __init__(self, width=8):
        super().__init__()
        self.encoders = nn.ModuleList([Conv(3, width)] +
                                     [nn.Sequential(Separable(a, b), Separable(b, b))
                                      for a, b in zip([width, width*2, width*4], [width*2, width*4, width*8])])
        self.decoders = nn.ModuleList([nn.Sequential(Separable(a+b, b), Separable(b, b))
                                      for a, b in zip([width*8, width*4, width*2], [width*4, width*2, width])])
        self.output = nn.Conv2d(width, 1, 1)

    def forward(self, image):
        features = []
        x = image
        for i, block in enumerate(self.encoders):
            x = block(F.max_pool2d(x, 2) if i else x)
            features.append(x)
        for block, skip in zip(self.decoders, features[-2::-1]):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = block(torch.cat([skip, x], dim=1))
        return self.output(x)[:, 0]


class FCNLite(nn.Module):
    """Stride-8 CNN with additive stride-2/4/8 logit fusion; no VGG pretraining."""
    def __init__(self, width=8):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(Conv(3, width, 2), Conv(width, width)),
            nn.Sequential(Conv(width, width*2, 2), Conv(width*2, width*2)),
            nn.Sequential(Conv(width*2, width*4, 2), Conv(width*4, width*4)),
        ])
        self.heads = nn.ModuleList([nn.Conv2d(w, 1, 1) for w in [width, width*2, width*4]])

    def forward(self, image):
        x, logits = image, []
        for stage, head in zip(self.stages, self.heads):
            x = stage(x)
            logits.append(F.interpolate(head(x), size=image.shape[-2:], mode='bilinear', align_corners=False))
        return sum(logits)[:, 0]


def build_model(name, width=8):
    if name == 'unet':
        return UNet(width, 'group')
    if name == 'ds_unet':
        return DSUNet(width)
    if name == 'fcn_lite':
        return FCNLite(width)
    raise ValueError(f'Unknown model: {name}')
