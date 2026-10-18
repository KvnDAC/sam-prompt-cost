"""U-Net baseline with configurable width; RGB input, one binary-mask logit."""
import math
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, incoming, outgoing, normalization):
        blocks = []
        for channels in (incoming, outgoing):
            blocks.append(nn.Conv2d(channels, outgoing, 3, padding=1))
            if normalization == "group":
                blocks.append(nn.GroupNorm(math.gcd(8, outgoing), outgoing))
            blocks.append(nn.ReLU(inplace=True))
        super().__init__(*blocks)


class UNet(nn.Module):
    """Four down/up stages, padded convolutions, no pretrained weights or batch norm.

    This is a declared U-Net variant, not an exact reproduction of the 2015 recipe.
    """
    def __init__(self, base_channels=32, normalization="none"):
        super().__init__()
        if normalization not in ("none", "group"):
            raise ValueError(normalization)
        widths = [base_channels * 2**i for i in range(5)]
        self.encoders = nn.ModuleList([DoubleConv(3, widths[0], normalization)] +
                                     [DoubleConv(a, b, normalization) for a, b in zip(widths, widths[1:])])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(a, b, 2, stride=2)
                                 for a, b in zip(widths[:0:-1], widths[-2::-1])])
        self.decoders = nn.ModuleList([DoubleConv(2 * w, w, normalization) for w in widths[-2::-1]])
        self.output = nn.Conv2d(widths[0], 1, 1)

    def forward(self, image):
        skips = []
        x = image
        for i, encoder in enumerate(self.encoders):
            if i:
                x = F.max_pool2d(x, 2)
            x = encoder(x)
            skips.append(x)
        for up, decoder, skip in zip(self.ups, self.decoders, skips[-2::-1]):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([skip, x], dim=1))
        return self.output(x)[:, 0]


def prepare_arrays(image, mask, size):
    """Preserve aspect ratio, top-left placement; mask/valid exclude padded pixels."""
    if image.shape[:2] != mask.shape or size < 16:
        raise ValueError("Image/mask dimensions must agree and size must be >=16")
    h, w = mask.shape
    scale = size / max(h, w)
    hh, ww = max(1, round(h * scale)), max(1, round(w * scale))
    rgb = np.asarray(Image.fromarray(image).resize((ww, hh), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    gt = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((ww, hh), Image.Resampling.NEAREST)) > 127
    tensor = torch.zeros(3, size, size)
    target, valid = torch.zeros(size, size), torch.zeros(size, size)
    tensor[:, :hh, :ww] = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
    target[:hh, :ww] = torch.from_numpy(gt.copy()).float()
    valid[:hh, :ww] = 1
    return tensor, target, valid, (hh, ww)


def restore_logits(logits, input_size, original_size):
    """Remove padding before returning to original image resolution."""
    h, w = input_size
    return F.interpolate(logits[None, None, :h, :w], size=original_size,
                         mode="bilinear", align_corners=False)[0, 0]
